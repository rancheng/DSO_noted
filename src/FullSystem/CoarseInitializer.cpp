/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"


#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

    CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0, 0), thisToNext(SE3()) {
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            points[lvl] = 0;
            numPoints[lvl] = 0;
        }

        JbBuffer = new Vec10f[ww * hh]; // 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
        JbBuffer_new = new Vec10f[ww * hh];


        frameID = -1;
        fixAffine = true;
        printDebug = false;
        // wM is a diagonal matrix dump all the scale information
        wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
        wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
        wM.diagonal()[6] = SCALE_A;
        wM.diagonal()[7] = SCALE_B;
    }

    CoarseInitializer::~CoarseInitializer() {
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            if (points[lvl] != 0) delete[] points[lvl];
        }

        delete[] JbBuffer;
        delete[] JbBuffer_new;
    }


    bool CoarseInitializer::trackFrame(FrameHessian *newFrameHessian, std::vector<IOWrap::Output3DWrapper *> &wraps) {
        newFrame = newFrameHessian;
        // update the target frame into visualization wrapper.
        for (IOWrap::Output3DWrapper *ow : wraps)
            ow->pushLiveFrame(newFrameHessian);
        // maximum itarations for different scales.
        int maxIterations[] = {5, 5, 10, 30, 50};


        alphaK = 2.5 * 2.5;//*freeDebugParam1*freeDebugParam1;
        alphaW = 150 * 150;//*freeDebugParam2*freeDebugParam2;
        regWeight = 0.8;//*freeDebugParam4;
        couplingWeight = 1;//*freeDebugParam5;
        // if it's not snapped? what snapped mean? stored? successfully tracked?
        // these initialization steps shows that snapped means established a stable tacking in that frame.
        // ###########################
        // Now I know, snapped is a flag returned by the tracker, if tracker successfully locked this frame,
        // that means it was snapped, which is tracked. that's why if it's tracked, all idepth and hessian stuff
        // would be already available.
        if (!snapped) { // if not tracked or no tracking successful, initialize all selected point in this frame
            thisToNext.translation().setZero(); // initialize thisToNext transformation matrix SE3 to 0
            for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) // pyrLevelsUsed is 6 in settings.h
            {
                int npts = numPoints[lvl];
                Pnt *ptsl = points[lvl];
                for (int i = 0; i < npts; i++) { // loop all selected points and initialize their iR idepth and hessian member variables.
                    ptsl[i].iR = 1; // what is iR? what R means?
                    ptsl[i].idepth_new = 1; // this idepth_new should be a dummy variable store the idepth estimation
                    ptsl[i].lastHessian = 0; // another dummy variable to store hessian
                }
            }
        }


        SE3 refToNew_current = thisToNext; // if not snapped, this is all 0
        AffLight refToNew_aff_current = thisToNext_aff; // initialized as (0,0) in constructor.
        // firstFrame was set in the CoarseInitializer::setFirst function
        // newFrame is the current tracking frame.
        // this if statement lookup whether the frames contains affine exposure model
        if (firstFrame->ab_exposure > 0 && newFrame->ab_exposure > 0)
            refToNew_aff_current = AffLight(logf(newFrame->ab_exposure / firstFrame->ab_exposure),
                                            0); // coarse approximation.


        Vec3f latestRes = Vec3f::Zero();
        // hmm... interesting, this loop start from smaller scale side
        // so, by doing this inverse order loop, the globally stable points are selected to loop out the
        // coarse SE3 matrix in a global manner, and will be kept refined in the loop on the larger scales.
        for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--) { //start from lvl=5... which is the 6th scale in the index..


            if (lvl < pyrLevelsUsed - 1)
                propagateDown(lvl + 1); // now I know why they loop from smaller scale to larger scale, they wanna propogate down....
            // now this iR is updated for each point in each lvl.
            Mat88f H, Hsc; // H and Hsc are 8 by 8 matrix, what are they?
            Vec8f b, bsc; // what does sc mean?
            resetPoints(lvl); // normalize the idepth of each points in that lvl (make it gradient friendly)
            Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
            applyStep(lvl);

            float lambda = 0.1;
            float eps = 1e-4;
            int fails = 0;

            if (printDebug) {
                printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
                       lvl, 0, lambda,
                       "INITIA",
                       sqrtf((float) (resOld[0] / resOld[2])),
                       sqrtf((float) (resOld[1] / resOld[2])),
                       sqrtf((float) (resOld[0] / resOld[2])),
                       sqrtf((float) (resOld[1] / resOld[2])),
                       (resOld[0] + resOld[1]) / resOld[2],
                       (resOld[0] + resOld[1]) / resOld[2],
                       0.0f);
                std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose()
                          << "\n";
            }

            int iteration = 0;
            while (true) {
                Mat88f Hl = H;
                for (int i = 0; i < 8; i++) Hl(i, i) *= (1 + lambda);
                Hl -= Hsc * (1 / (1 + lambda));
                Vec8f bl = b - bsc * (1 / (1 + lambda));

                Hl = wM * Hl * wM * (0.01f / (w[lvl] * h[lvl]));
                bl = wM * bl * (0.01f / (w[lvl] * h[lvl]));


                Vec8f inc;
                if (fixAffine) {
                    inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6, 6>() *
                                      (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
                    inc.tail<2>().setZero();
                } else
                    inc = -(wM * (Hl.ldlt().solve(bl)));    //=-H^-1 * b.


                SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
                AffLight refToNew_aff_new = refToNew_aff_current;
                refToNew_aff_new.a += inc[6];
                refToNew_aff_new.b += inc[7];
                doStep(lvl, lambda, inc);


                Mat88f H_new, Hsc_new;
                Vec8f b_new, bsc_new;
                Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
                Vec3f regEnergy = calcEC(lvl);

                float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
                float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);


                bool accept = eTotalOld > eTotalNew;

                if (printDebug) {
                    printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
                           lvl, iteration, lambda,
                           (accept ? "ACCEPT" : "REJECT"),
                           sqrtf((float) (resOld[0] / resOld[2])),
                           sqrtf((float) (regEnergy[0] / regEnergy[2])),
                           sqrtf((float) (resOld[1] / resOld[2])),
                           sqrtf((float) (resNew[0] / resNew[2])),
                           sqrtf((float) (regEnergy[1] / regEnergy[2])),
                           sqrtf((float) (resNew[1] / resNew[2])),
                           eTotalOld / resNew[2],
                           eTotalNew / resNew[2],
                           inc.norm());
                    std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose()
                              << "\n";
                }

                if (accept) {

                    if (resNew[1] == alphaK * numPoints[lvl])
                        snapped = true;
                    H = H_new;
                    b = b_new;
                    Hsc = Hsc_new;
                    bsc = bsc_new;
                    resOld = resNew;
                    refToNew_aff_current = refToNew_aff_new;
                    refToNew_current = refToNew_new;
                    applyStep(lvl);
                    optReg(lvl);
                    lambda *= 0.5;
                    fails = 0;
                    if (lambda < 0.0001) lambda = 0.0001;
                } else {
                    fails++;
                    lambda *= 4;
                    if (lambda > 10000) lambda = 10000;
                }

                bool quitOpt = false;

                if (!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2) {
                    Mat88f H, Hsc;
                    Vec8f b, bsc;

                    quitOpt = true;
                }


                if (quitOpt) break;
                iteration++;
            }
            latestRes = resOld;

        }


        thisToNext = refToNew_current;
        thisToNext_aff = refToNew_aff_current;

        for (int i = 0; i < pyrLevelsUsed - 1; i++)
            propagateUp(i);


        frameID++;
        if (!snapped) snappedAt = 0;

        if (snapped && snappedAt == 0)
            snappedAt = frameID;


        debugPlot(0, wraps);


        return snapped && frameID > snappedAt + 5;
    }

    void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper *> &wraps) {
        bool needCall = false;
        for (IOWrap::Output3DWrapper *ow : wraps)
            needCall = needCall || ow->needPushDepthImage();
        if (!needCall) return;


        int wl = w[lvl], hl = h[lvl];
        Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];

        MinimalImageB3 iRImg(wl, hl);

        for (int i = 0; i < wl * hl; i++) // this is just grey scale img
            iRImg.at(i) = Vec3b(colorRef[i][0], colorRef[i][0], colorRef[i][0]);


        int npts = numPoints[lvl];

        float nid = 0, sid = 0;
        for (int i = 0; i < npts; i++) {
            Pnt *point = points[lvl] + i;
            if (point->isGood) {
                nid++; // number of tracked point, or idepth aggregated
                sid += point->iR; // this is sum of point idepth regression
            }
        }
        float fac = nid / sid; // a normalize factor? for visualization effect?


        for (int i = 0; i < npts; i++) {
            Pnt *point = points[lvl] + i;

            if (!point->isGood)
                iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, Vec3b(0, 0, 0));

            else
                iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, makeRainbow3B(point->iR * fac));
        }


        //IOWrap::displayImage("idepth-R", &iRImg, false);
        for (IOWrap::Output3DWrapper *ow : wraps)
            ow->pushDepthImage(&iRImg);
    }

// calculates residual, Hessian and Hessian-block needed for re-substituting depth.
    Vec3f CoarseInitializer::calcResAndGS(
            int lvl, Mat88f &H_out, Vec8f &b_out,
            Mat88f &H_out_sc, Vec8f &b_out_sc,
            const SE3 &refToNew, AffLight refToNew_aff,
            bool plot) {
        int wl = w[lvl], hl = h[lvl]; // width and height in certain lvl.
        Eigen::Vector3f *colorRef = firstFrame->dIp[lvl]; // dIp is a 3xn matrix. which contains color, dx, dy
        Eigen::Vector3f *colorNew = newFrame->dIp[lvl]; // same as colorNew, you can see that Vector3f, newFrame was referenced in trackFrame function.

        Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>(); // rotation matrix that convert from host frame to target frame.
        Vec3f t = refToNew.translation().cast<float>(); // original t move point from host frame to target frame.
        Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b); // affine model

        // intrinsic, lvl specific.
        float fxl = fx[lvl];
        float fyl = fy[lvl];
        float cxl = cx[lvl];
        float cyl = cy[lvl];

        // defined a set of accumulator to upate H and b.
        Accumulator11 E;
        acc9.initialize();
        E.initialize();

        // loop all selected points in this lvl.
        int npts = numPoints[lvl];
        Pnt *ptsl = points[lvl];
        for (int i = 0; i < npts; i++) {

            Pnt *point = ptsl + i;

            point->maxstep = 1e10;
            if (!point->isGood) {
                // initialize bad points.
                E.updateSingle((float) (point->energy[0])); // E[0] += UenergyPhotometric
                point->energy_new = point->energy;
                point->isGood_new = false;
                continue;
            }

            VecNRf dp0; //Vec8f... why this is 8f? well, look at the for loop below, it's storing dp0 of 8 directions.
            VecNRf dp1;
            VecNRf dp2;
            VecNRf dp3;
            VecNRf dp4;
            VecNRf dp5;
            VecNRf dp6;
            VecNRf dp7;
            VecNRf dd;
            VecNRf r;
            JbBuffer_new[i].setZero(); //JbBuffer_new content: 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.

            // sum over all residuals.
            bool isGood = true;
            float energy = 0;
            // for each selected point in lvl. loop eight directions:
            for (int idx = 0; idx < patternNum; idx++) {
                int dx = patternP[idx][0]; // dx dy in 8 directions.
                int dy = patternP[idx][1];

                // projected point with the offset directions.
                // this is RKi * x + t*idepth = X in target frame. pt[2] = X[2] = 1 + t[2] * idepth
                Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) + t * point->idepth_new;
                float u = pt[0] / pt[2]; // project to the target frame's image plane.
                float v = pt[1] / pt[2];
                float Ku = fxl * u + cxl;
                float Kv = fyl * v + cyl;
                float new_idepth = point->idepth_new / pt[2]; // idepth_new is the estimated z, and pt[2] is projected z in new frame.

                if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && new_idepth > 0)) {
                    isGood = false;
                    break;
                }

                Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
                //Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

                //float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
                float rlR = getInterpolatedElement31(colorRef, point->u + dx, point->v + dy, wl);

                if (!std::isfinite(rlR) || !std::isfinite((float) hitColor[0])) {
                    isGood = false;
                    break;
                }


                float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
                float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
                energy += hw * residual * residual * (2 - hw);

                // t is translation matrix. t[0] is x, t[1] is y, and t[2] is z
                // t[0] - t[2] * u = x - z*u
                // dxdd = (x-zu)/(1+z*idepth)
                float dxdd = (t[0] - t[2] * u) / pt[2];
                float dydd = (t[1] - t[2] * v) / pt[2];

                if (hw < 1) hw = sqrtf(hw);
                float dxInterp = hw * hitColor[1] * fxl;
                float dyInterp = hw * hitColor[2] * fyl;
                dp0[idx] = new_idepth * dxInterp;
                dp1[idx] = new_idepth * dyInterp;
                dp2[idx] = -new_idepth * (u * dxInterp + v * dyInterp);
                dp3[idx] = -u * v * dxInterp - (1 + v * v) * dyInterp;
                dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp;
                dp5[idx] = -v * dxInterp + u * dyInterp;
                dp6[idx] = -hw * r2new_aff[0] * rlR;
                dp7[idx] = -hw * 1;
                dd[idx] = dxInterp * dxdd + dyInterp * dydd;
                r[idx] = hw * residual;

                float maxstep = 1.0f / Vec2f(dxdd * fxl, dydd * fyl).norm();
                if (maxstep < point->maxstep) point->maxstep = maxstep;

                // immediately compute dp*dd' and dd*dd' in JbBuffer1.
                JbBuffer_new[i][0] += dp0[idx] * dd[idx];
                JbBuffer_new[i][1] += dp1[idx] * dd[idx];
                JbBuffer_new[i][2] += dp2[idx] * dd[idx];
                JbBuffer_new[i][3] += dp3[idx] * dd[idx];
                JbBuffer_new[i][4] += dp4[idx] * dd[idx];
                JbBuffer_new[i][5] += dp5[idx] * dd[idx];
                JbBuffer_new[i][6] += dp6[idx] * dd[idx];
                JbBuffer_new[i][7] += dp7[idx] * dd[idx];
                JbBuffer_new[i][8] += r[idx] * dd[idx];
                JbBuffer_new[i][9] += dd[idx] * dd[idx];
            }

            if (!isGood || energy > point->outlierTH * 20) {
                E.updateSingle((float) (point->energy[0]));
                point->isGood_new = false;
                point->energy_new = point->energy;
                continue;
            }


            // add into energy.
            E.updateSingle(energy);
            point->isGood_new = true;
            point->energy_new[0] = energy;

            // update Hessian matrix.
            for (int i = 0; i + 3 < patternNum; i += 4)
                acc9.updateSSE(
                        _mm_load_ps(((float *) (&dp0)) + i),
                        _mm_load_ps(((float *) (&dp1)) + i),
                        _mm_load_ps(((float *) (&dp2)) + i),
                        _mm_load_ps(((float *) (&dp3)) + i),
                        _mm_load_ps(((float *) (&dp4)) + i),
                        _mm_load_ps(((float *) (&dp5)) + i),
                        _mm_load_ps(((float *) (&dp6)) + i),
                        _mm_load_ps(((float *) (&dp7)) + i),
                        _mm_load_ps(((float *) (&r)) + i));


            for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
                acc9.updateSingle(
                        (float) dp0[i], (float) dp1[i], (float) dp2[i], (float) dp3[i],
                        (float) dp4[i], (float) dp5[i], (float) dp6[i], (float) dp7[i],
                        (float) r[i]);


        }

        E.finish();
        acc9.finish();






        // calculate alpha energy, and decide if we cap it.
        Accumulator11 EAlpha;
        EAlpha.initialize();
        for (int i = 0; i < npts; i++) {
            Pnt *point = ptsl + i;
            if (!point->isGood_new) {
                E.updateSingle((float) (point->energy[1]));
            } else {
                point->energy_new[1] = (point->idepth_new - 1) * (point->idepth_new - 1);
                E.updateSingle((float) (point->energy_new[1]));
            }
        }
        EAlpha.finish();
        float alphaEnergy = alphaW * (EAlpha.A + refToNew.translation().squaredNorm() * npts);

        //printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);


        // compute alpha opt.
        float alphaOpt;
        if (alphaEnergy > alphaK * npts) {
            alphaOpt = 0;
            alphaEnergy = alphaK * npts;
        } else {
            alphaOpt = alphaW;
        }


        acc9SC.initialize();
        for (int i = 0; i < npts; i++) {
            Pnt *point = ptsl + i;
            if (!point->isGood_new)
                continue;

            point->lastHessian_new = JbBuffer_new[i][9]; // JbBuffer_new[i][9] is dd*dd which is squared second derivative, the hessian entry

            JbBuffer_new[i][8] += alphaOpt * (point->idepth_new - 1);
            JbBuffer_new[i][9] += alphaOpt;

            if (alphaOpt == 0) {
                JbBuffer_new[i][8] += couplingWeight * (point->idepth_new - point->iR);
                JbBuffer_new[i][9] += couplingWeight;
            }

            JbBuffer_new[i][9] = 1 / (1 + JbBuffer_new[i][9]);
            acc9SC.updateSingleWeighted(
                    (float) JbBuffer_new[i][0], (float) JbBuffer_new[i][1], (float) JbBuffer_new[i][2],
                    (float) JbBuffer_new[i][3],
                    (float) JbBuffer_new[i][4], (float) JbBuffer_new[i][5], (float) JbBuffer_new[i][6],
                    (float) JbBuffer_new[i][7],
                    (float) JbBuffer_new[i][8], (float) JbBuffer_new[i][9]);
        }
        acc9SC.finish();


        //printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
        H_out = acc9.H.topLeftCorner<8, 8>();// / acc9.num;
        b_out = acc9.H.topRightCorner<8, 1>();// / acc9.num;
        H_out_sc = acc9SC.H.topLeftCorner<8, 8>();// / acc9.num;
        b_out_sc = acc9SC.H.topRightCorner<8, 1>();// / acc9.num;



        H_out(0, 0) += alphaOpt * npts;
        H_out(1, 1) += alphaOpt * npts;
        H_out(2, 2) += alphaOpt * npts;

        Vec3f tlog = refToNew.log().head<3>().cast<float>();
        b_out[0] += tlog[0] * alphaOpt * npts;
        b_out[1] += tlog[1] * alphaOpt * npts;
        b_out[2] += tlog[2] * alphaOpt * npts;


        return Vec3f(E.A, alphaEnergy, E.num);
    }

    float CoarseInitializer::rescale() {
        float factor = 20 * thisToNext.translation().norm();
//	float factori = 1.0f/factor;
//	float factori2 = factori*factori;
//
//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
//	{
//		int npts = numPoints[lvl];
//		Pnt* ptsl = points[lvl];
//		for(int i=0;i<npts;i++)
//		{
//			ptsl[i].iR *= factor;
//			ptsl[i].idepth_new *= factor;
//			ptsl[i].lastHessian *= factori2;
//		}
//	}
//	thisToNext.translation() *= factori;

        return factor;
    }


    Vec3f CoarseInitializer::calcEC(int lvl) {
        if (!snapped) return Vec3f(0, 0, numPoints[lvl]);
        AccumulatorX<2> E;
        E.initialize();
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++) {
            Pnt *point = points[lvl] + i;
            if (!point->isGood_new) continue;
            float rOld = (point->idepth - point->iR);
            float rNew = (point->idepth_new - point->iR);
            E.updateNoWeight(Vec2f(rOld * rOld, rNew * rNew));

            //printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
        }
        E.finish();

        //printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
        return Vec3f(couplingWeight * E.A1m[0], couplingWeight * E.A1m[1], E.num);
    }

    // idepth regression (use the median) of each point by it's well-tracked nearest neighbours
    void CoarseInitializer::optReg(int lvl) {
        int npts = numPoints[lvl];
        Pnt *ptsl = points[lvl];
        if (!snapped) { // if not tracked, return... this optReg is only for tracked frame. that means you have already finished track function.
            for (int i = 0; i < npts; i++)
                ptsl[i].iR = 1; // initialize inverse depth regression for every point.
            return;
        }


        for (int i = 0; i < npts; i++) {
            Pnt *point = ptsl + i;
            if (!point->isGood) continue; // skip all good points.

            float idnn[10]; // idepth regression, actually the iR of nearest neighbour
            int nnn = 0; // nearest neighbour number
            // collect the 10 nearest neighbour (if the are good points) inverse depth
            for (int j = 0; j < 10; j++) {
                if (point->neighbours[j] == -1) continue; //skip uninitialized neighbours
                // note that neighbours is a list size of 10
                // remember in flann, neighbours list was storing the neighour points'
                // index in original pts list, thus this ptsl + point->neighbours[j]
                // can reach to the neighbour point.
                Pnt *other = ptsl + point->neighbours[j];
                if (!other->isGood) continue; // again, skip all non-tracking neighbour.
                idnn[nnn] = other->iR; // this idnn record all the  good neighbour idepth regression.
                nnn++;
            }
            // more than two good neighbours will trigger idepth regression. (actually normalize the idepth estimation with nearby points)
            if (nnn > 2) {
                std::nth_element(idnn, idnn + nnn / 2, idnn + nnn); // find the median of idnn...
                point->iR = (1 - regWeight) * point->idepth + regWeight * idnn[nnn / 2]; // 0.2 * point->idepth + 0.8*median_idnn...
            }
        }

    }


    void CoarseInitializer::propagateUp(int srcLvl) {
        assert(srcLvl + 1 < pyrLevelsUsed);
        // set idepth of target

        int nptss = numPoints[srcLvl];
        int nptst = numPoints[srcLvl + 1];
        Pnt *ptss = points[srcLvl];
        Pnt *ptst = points[srcLvl + 1];

        // set to zero.
        for (int i = 0; i < nptst; i++) {
            Pnt *parent = ptst + i;
            parent->iR = 0;
            parent->iRSumNum = 0;
        }

        for (int i = 0; i < nptss; i++) {
            Pnt *point = ptss + i;
            if (!point->isGood) continue;

            Pnt *parent = ptst + point->parent;
            parent->iR += point->iR * point->lastHessian;
            parent->iRSumNum += point->lastHessian;
        }

        for (int i = 0; i < nptst; i++) {
            Pnt *parent = ptst + i;
            if (parent->iRSumNum > 0) {
                parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
                parent->isGood = true;
            }
        }

        optReg(srcLvl + 1);
    }
    // normalize the idepth of point by the parent idepth with corresponding point hessians.
    void CoarseInitializer::propagateDown(int srcLvl) {
        assert(srcLvl > 0);
        // set idepth of target
        // here pts is the selected points by pixelSelector...
        int nptst = numPoints[srcLvl - 1]; // number of pts in target scale.
        Pnt *ptss = points[srcLvl]; // ptss is points source, points from smaller scale (upper layer)
        Pnt *ptst = points[srcLvl - 1]; // ptst is points target, points from larger scale (bottom layer)

        for (int i = 0; i < nptst; i++) {
            Pnt *point = ptst + i; // point is looking to each ptst index.
            Pnt *parent = ptss + point->parent; // parent is the idx of the upper layer closest point.

            if (!parent->isGood || parent->lastHessian < 0.1) continue; // skip bad parent point
            if (!point->isGood) { // initialize the current point in the target scale.
                point->iR = point->idepth = point->idepth_new = parent->iR; // use parent idepth as the current layer's idepth
                point->isGood = true; // yes, you have got a valid parent idepth, certainly you should be a good point.
                point->lastHessian = 0; // lastHessian was updated by lastHessian_New, and lastHessian_New is updated from JbBuffer[idx][9] which is dd[idx]*dd[idx].
            } else {
                float newiR = (point->iR * point->lastHessian * 2 + parent->iR * parent->lastHessian) /
                              (point->lastHessian * 2 + parent->lastHessian); // so this newiR is a normalized iR from current point and parent point with the corresponding hessian matrix.
                point->iR = point->idepth = point->idepth_new = newiR; // now this normalized newiR updated iR, idepth and idepth_new...
            }
        }
        optReg(srcLvl - 1);
    }

// this function loops through all scales and use average pooling to create scale spaces and calculate dx dy for each scale.
    void CoarseInitializer::makeGradients(Eigen::Vector3f **data) {
        for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
            int lvlm1 = lvl - 1;
            int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

            Eigen::Vector3f *dINew_l = data[lvl]; // dINew_l is the scale from 1 to 8
            Eigen::Vector3f *dINew_lm = data[lvlm1]; // dINew_lm is the upper (larger) scale above dINew_l

            for (int y = 0; y < hl; y++)
                for (int x = 0; x < wl; x++)
                    // normalize the dINew_l pixel color at x, y by average the upper layer's 4 nearby pixel patch.
                    dINew_l[x + y * wl][0] = 0.25f * (dINew_lm[2 * x + 2 * y * wlm1][0] +
                                                      dINew_lm[2 * x + 1 + 2 * y * wlm1][0] +
                                                      dINew_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
                                                      dINew_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);

            for (int idx = wl; idx < wl * (hl - 1); idx++) {
                dINew_l[idx][1] = 0.5f * (dINew_l[idx + 1][0] - dINew_l[idx - 1][0]); // calculate dx
                dINew_l[idx][2] = 0.5f * (dINew_l[idx + wl][0] - dINew_l[idx - wl][0]); // calculate dy
            }
        }
    }

    // this function select all the points in different scale, one thing note-worthy is that all points in smaller scales
    // are selected with max gradient selector (in PixelSelector), which is faster than random gradient selector (in PixelSelector2)
    void CoarseInitializer::setFirst(CalibHessian *HCalib, FrameHessian *newFrameHessian) {

        makeK(HCalib);
        firstFrame = newFrameHessian; // first Frame in CoarseInitializer is equivalent to host frame in frame window.

        PixelSelector sel(w[0],
                          h[0]); // basically this selector select points in different scales that gradient collinear with the random direction setup.

        float *statusMap = new float[w[0] *
                                     h[0]]; // statusMap is the point select map. record which point are selected from which scale.
        bool *statusMapB = new bool[w[0] *
                                    h[0]]; // statusMapB here B is binary, see this is a bool map. record weather a point should be select or not.

        float densities[] = {0.03, 0.05, 0.15, 0.5,
                             1}; // I see, this is density for sample: 3% on largest scale. and 100% on smallest scale.
        // float numWant = density; this is in PixelSelector2.cpp. which is the point number makeMaps want.
        // loop through all level of scales
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            sel.currentPotential = 3; // okay, start from largest potential patch size, and will reach to 1 in pixel selector.
            int npts;
            if (lvl == 0)
                npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * w[0] * h[0], 1, false,
                                    2); // selected points in random directions.
            else
                npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl],
                                       densities[lvl] * w[0] * h[0]); // select points in max gradient direction...
            // npts is the number of points selected.

            if (points[lvl] != 0) delete[] points[lvl]; // clear the memory
            points[lvl] = new Pnt[npts]; // allocate new slots to store the selected points in lvl.

            // set idepth map to initially 1 everywhere.
            int wl = w[lvl], hl = h[lvl]; // w h in lvl scale.
            Pnt *pl = points[lvl]; // pointer to get the points in lvl.
            int nl = 0;
            // this whole loop covers the whole image space with in the padding, in scale lvl.
            for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++)
                for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++) {
                    //if(x==2) printf("y=%d!\n",y);
                    // remember that statusMapB is the binary map of the point selected.
                    // statusMap is the hash map for the selected points in different level.
                    // this condition is (not in first level and the point in x,y location was selected)
                    // or it's first level but the selected point is not from first level.
                    if ((lvl != 0 && statusMapB[x + y * wl]) || (lvl == 0 && statusMap[x + y * wl] != 0)) {
                        // initialize and assign the content of selected point in lvl
                        //assert(patternNum==9);
                        pl[nl].u = x + 0.1;
                        pl[nl].v = y + 0.1;
                        pl[nl].idepth = 1;
                        pl[nl].iR = 1;
                        pl[nl].isGood = true;
                        pl[nl].energy.setZero();
                        pl[nl].lastHessian = 0;
                        pl[nl].lastHessian_new = 0;
                        pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];
                        // color dx, dy of this point.
                        Eigen::Vector3f *cpt = firstFrame->dIp[lvl] + x + y * w[lvl];
                        float sumGrad2 = 0;
                        // now aggregate the gradients in 8 directions.
                        for (int idx = 0; idx < patternNum; idx++) {
                            int dx = patternP[idx][0];
                            int dy = patternP[idx][1];
                            float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
                            sumGrad2 += absgrad; // this sumGrad2 was intent to control the outlier threshold for each point.
                        }

//				float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
//				pl[nl].outlierTH = patternNum*gth*gth;
//

                        pl[nl].outlierTH =
                                patternNum * setting_outlierTH; // but they decide to use the constant threshold.



                        nl++; // nl is the final selected points in the given level.
                        assert(nl <= npts);
                    }
                }

            // now nl is the final point selected in scale lvl.
            numPoints[lvl] = nl; // numPoints record the number of chosen point in different lvl.
        }
        // --------end of lvl loop.-------------


        delete[] statusMap; // since all points are stored in pl.
        delete[] statusMapB; // status map (selection map) has finished their purpose.

        makeNN(); // build kdtree of selected points in each lvl and find nearest neighbours in same lvl and parent lvl (smaller scaled layer)

        thisToNext = SE3(); // create an empty transformation matrix.
        snapped = false; // don't know the purpose of this yet. snap means store it by flash.

        frameID = snappedAt = 0; // hmm... interesting, does this means snap will store the frame in memory?

        for (int i = 0; i < pyrLevelsUsed; i++)
            dGrads[i].setZero(); // dGrads is a vect3f... list of vector.  seems like they have never used this dGrads.

    }
    // normalize the idepth of all points in that level with it's neighbours.
    void CoarseInitializer::resetPoints(int lvl) {
        Pnt *pts = points[lvl]; // get selected points in that lvl.
        int npts = numPoints[lvl]; // get the selected point number in that lvl.
        for (int i = 0; i < npts; i++) {
            pts[i].energy.setZero(); // energy reset
            pts[i].idepth_new = pts[i].idepth; // idepth_new dump.

            // if lvl == 5 and the point i is not good.
            if (lvl == pyrLevelsUsed - 1 && !pts[i].isGood) {
                float snd = 0, sn = 0;
                for (int n = 0; n < 10; n++) {
                    if (pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue; // if no neighbor or neighbor is not good
                    snd += pts[pts[i].neighbours[n]].iR; // aggregate the inverse re-substitude depth of good neighbours
                    sn += 1; // count the aggregate times to do averaging later on.
                }

                if (sn > 0) {
                    pts[i].isGood = true; // if there's some good neighbour, let the neighbour's depth as the current point depth.
                    pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd / sn; // iR is the avg(sum(neighbour.iR))
                }
            }
        }
    }

    void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc) {

        const float maxPixelStep = 0.25;
        const float idMaxStep = 1e10;
        Pnt *pts = points[lvl];
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++) {
            if (!pts[i].isGood) continue;


            float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
            float step = -b * JbBuffer[i][9] / (1 + lambda);


            float maxstep = maxPixelStep * pts[i].maxstep;
            if (maxstep > idMaxStep) maxstep = idMaxStep;

            if (step > maxstep) step = maxstep;
            if (step < -maxstep) step = -maxstep;

            float newIdepth = pts[i].idepth + step;
            if (newIdepth < 1e-3) newIdepth = 1e-3;
            if (newIdepth > 50) newIdepth = 50;
            pts[i].idepth_new = newIdepth;
        }

    }

    void CoarseInitializer::applyStep(int lvl) {
        Pnt *pts = points[lvl];
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++) {
            if (!pts[i].isGood) {
                pts[i].idepth = pts[i].idepth_new = pts[i].iR;
                continue;
            }
            pts[i].energy = pts[i].energy_new;
            pts[i].isGood = pts[i].isGood_new;
            pts[i].idepth = pts[i].idepth_new;
            pts[i].lastHessian = pts[i].lastHessian_new;
        }
        std::swap<Vec10f *>(JbBuffer, JbBuffer_new);
    }

    void CoarseInitializer::makeK(CalibHessian *HCalib) {
        w[0] = wG[0];
        h[0] = hG[0];

        fx[0] = HCalib->fxl();
        fy[0] = HCalib->fyl();
        cx[0] = HCalib->cxl();
        cy[0] = HCalib->cyl();

        for (int level = 1; level < pyrLevelsUsed; ++level) {
            w[level] = w[0] >> level;
            h[level] = h[0] >> level;
            fx[level] = fx[level - 1] * 0.5;
            fy[level] = fy[level - 1] * 0.5;
            cx[level] = (cx[0] + 0.5) / ((int) 1 << level) - 0.5;
            cy[level] = (cy[0] + 0.5) / ((int) 1 << level) - 0.5;
        }

        for (int level = 0; level < pyrLevelsUsed; ++level) {
            K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
            Ki[level] = K[level].inverse();
            fxi[level] = Ki[level](0, 0);
            fyi[level] = Ki[level](1, 1);
            cxi[level] = Ki[level](0, 2);
            cyi[level] = Ki[level](1, 2);
        }
    }

    // find the nearest neighbor of selected points (in the smaller scale space) from the it's larger parent scale space.
    // detailed original code can be found from:
    // https://github.com/jlblancoc/nanoflann/blob/master/examples/pointcloud_example.cpp
    // which is the author's repo of nanoflann
    void CoarseInitializer::makeNN() {
        const float NNDistFactor = 0.05;
        // construct a kd-tree index:
        // idex take 4 type of templates as parameter: 1. the distance class, 2. the data-source class 3. dimension (default -1), 4. index type (default size_t is type returned by the sizeof operator).
        // so this KDTree index is a 2d
        typedef nanoflann::KDTreeSingleIndexAdaptor<
                nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud>,
                FLANNPointcloud, 2> KDTree;

        // build indices
        FLANNPointcloud pcs[PYR_LEVELS];
        KDTree *indexes[PYR_LEVELS]; // create kd tree for each scale level.
        // now loop all scale spaces and build 2d kdtree from the selected points.
        for (int i = 0; i < pyrLevelsUsed; i++) {
            pcs[i] = FLANNPointcloud(numPoints[i], points[i]); // create the point cloud for each scale level.
            // KDTree constructor takes 3 parameters: dimensionality, inputData and params. this parameter sets the _leaf_max_size as 5
            indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5)); // create the 2d kd tree out of each point cloud.
            indexes[i]->buildIndex(); // initialize the kdtree index adaptor class.
        }

        const int nn = 10;

        // find NN & parents
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) { // why this is just pyrLevelsUsed? not pyrLevelsUsed - 1? where's parent?
            Pnt *pts = points[lvl];
            int npts = numPoints[lvl];

            int ret_index[nn]; // find nearest 10 parents.
            float ret_dist[nn];
            nanoflann::KNNResultSet<float, int, int> resultSet(nn);
            nanoflann::KNNResultSet<float, int, int> resultSet1(1);
            // loop all the selected points in the scale level
            for (int i = 0; i < npts; i++) {
                //resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
                resultSet.init(ret_index, ret_dist);
                Vec2f pt = Vec2f(pts[i].u, pts[i].v);
                // so find neighbors func takes three parameters: result_set, query_point and search_parameters
                // this function will put the search results into resultSet. which is a zipped data of (index, dist)
                indexes[lvl]->findNeighbors(resultSet, (float *) &pt, nanoflann::SearchParams());

                int myidx = 0;
                float sumDF = 0;
                for (int k = 0; k < nn; k++) { // this loop loops the nearest 10 points of pts[i] and dump their expf distance function
                    pts[i].neighbours[myidx] = ret_index[k]; // note that this ret_index is the point index in list pts[]
                    float df = expf(-ret_dist[k] * NNDistFactor); // small number inversely proportional to distance [0, 1]
                    sumDF += df; // sumDF aggregate the 10 nearest distance float, [0,10]
                    pts[i].neighboursDist[myidx] = df;
                    assert(ret_index[k] >= 0 && ret_index[k] < npts);
                    myidx++;
                }
                for (int k = 0; k < nn; k++)
                    pts[i].neighboursDist[k] *= 10 / sumDF; // the smaller the distance is, the smaller the neighbourdist[k] is, this is exponientially propotional to the distance found in kdtree neighbours.


                if (lvl < pyrLevelsUsed - 1) { // this if statement explained my question above, this is prepared for parent.
                    // remember this lvl is growing larger, which means the scale is smaller.
                    resultSet1.init(ret_index, ret_dist); // here they reuse the ret_index and ret_dist?
                    pt = pt * 0.5f - Vec2f(0.25f, 0.25f); // so parent is from the smaller scale.
                    // note that this pt is still in the current lvl, the 'parent' point is merely the index location change.
                    indexes[lvl + 1]->findNeighbors(resultSet1, (float *) &pt, nanoflann::SearchParams()); // this indexes is the kdtree in the smaller level.

                    pts[i].parent = ret_index[0]; // the nearest parent
                    pts[i].parentDist = expf(-ret_dist[0] * NNDistFactor); // the distance of the nearest parent

                    assert(ret_index[0] >= 0 && ret_index[0] < numPoints[lvl + 1]);
                } else {
                    pts[i].parent = -1; // last layer doesn't have parent.
                    pts[i].parentDist = -1;
                }
            }
        }



        // done.

        for (int i = 0; i < pyrLevelsUsed; i++)
            delete indexes[i];
    }
}

