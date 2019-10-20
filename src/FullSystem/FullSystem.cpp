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

#include "FullSystem/FullSystem.h"

#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>

namespace dso {
    int FrameHessian::instanceCounter = 0;
    int PointHessian::instanceCounter = 0;
    int CalibHessian::instanceCounter = 0;


    FullSystem::FullSystem() {

        int retstat = 0;
        // log, no need to explain
        if (setting_logStuff) {

            retstat += system("rm -rf logs");
            retstat += system("mkdir logs");

            retstat += system("rm -rf mats");
            retstat += system("mkdir mats");

            calibLog = new std::ofstream();
            calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
            calibLog->precision(12);

            numsLog = new std::ofstream();
            numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
            numsLog->precision(10);

            coarseTrackingLog = new std::ofstream();
            coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
            coarseTrackingLog->precision(10);

            eigenAllLog = new std::ofstream();
            eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
            eigenAllLog->precision(10);

            eigenPLog = new std::ofstream();
            eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
            eigenPLog->precision(10);

            eigenALog = new std::ofstream();
            eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
            eigenALog->precision(10);

            DiagonalLog = new std::ofstream();
            DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
            DiagonalLog->precision(10);

            variancesLog = new std::ofstream();
            variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
            variancesLog->precision(10);


            nullspacesLog = new std::ofstream();
            nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
            nullspacesLog->precision(10);
        } else {
            nullspacesLog = 0;
            variancesLog = 0;
            DiagonalLog = 0;
            eigenALog = 0;
            eigenPLog = 0;
            eigenAllLog = 0;
            numsLog = 0;
            calibLog = 0;
        }

        assert(retstat != 293847);


        //initialize the pixel selector map, which is just the image itself
        selectionMap = new float[wG[0] * hG[0]];
        // coarse distance map is used to store reprojection points, TODO: need to verify
        coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
        // initialize the coarse tracker template, coarse tracker is used on initialize and re-localizing
        // mainly used as "coarseTracker->trackNewestCoarse"
        coarseTracker = new CoarseTracker(wG[0], hG[0]);
        coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
        coarseInitializer = new CoarseInitializer(wG[0],
                                                  hG[0]); // initialize JbBuffer for pose residual and affine prior
        pixelSelector = new PixelSelector(wG[0], hG[0]);

        statistics_lastNumOptIts = 0;
        statistics_numDroppedPoints = 0;
        statistics_numActivatedPoints = 0;
        statistics_numCreatedPoints = 0;
        statistics_numForceDroppedResBwd = 0;
        statistics_numForceDroppedResFwd = 0;
        statistics_numMargResFwd = 0;
        statistics_numMargResBwd = 0;

        lastCoarseRMSE.setConstant(100);

        currentMinActDist = 2;
        initialized = false;


        ef = new EnergyFunctional();
        ef->red = &this->treadReduce;

        isLost = false;
        initFailed = false;


        needNewKFAfter = -1;

        linearizeOperation = true;
        runMapping = true;
        mappingThread = boost::thread(&FullSystem::mappingLoop, this);
        lastRefStopID = 0;


        minIdJetVisDebug = -1;
        maxIdJetVisDebug = -1;
        minIdJetVisTracker = -1;
        maxIdJetVisTracker = -1;
    }

    FullSystem::~FullSystem() {
        blockUntilMappingIsFinished();

        if (setting_logStuff) {
            calibLog->close();
            delete calibLog;
            numsLog->close();
            delete numsLog;
            coarseTrackingLog->close();
            delete coarseTrackingLog;
            //errorsLog->close(); delete errorsLog;
            eigenAllLog->close();
            delete eigenAllLog;
            eigenPLog->close();
            delete eigenPLog;
            eigenALog->close();
            delete eigenALog;
            DiagonalLog->close();
            delete DiagonalLog;
            variancesLog->close();
            delete variancesLog;
            nullspacesLog->close();
            delete nullspacesLog;
        }

        delete[] selectionMap;

        for (FrameShell *s : allFrameHistory)
            delete s;
        for (FrameHessian *fh : unmappedTrackedFrames)
            delete fh;

        delete coarseDistanceMap;
        delete coarseTracker;
        delete coarseTracker_forNewKF;
        delete coarseInitializer;
        delete pixelSelector;
        delete ef;
    }

    void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH) {

    }
    // invoked in the first like of main function, which firstly create those gamma function for Hcalib
    void FullSystem::setGammaFunction(float *BInv) {
        if (BInv == 0) return;

        // copy BInv.
        memcpy(Hcalib.Binv, BInv, sizeof(float) * 256);


        // invert.
        for (int i = 1; i < 255; i++) {
            // find val, such that Binv[val] = i.
            // I dont care about speed for this, so do it the stupid way.

            for (int s = 1; s < 255; s++) {
                if (BInv[s] <= i && BInv[s + 1] >= i) {
                    Hcalib.B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
                    break;
                }
            }
        }
        Hcalib.B[0] = 0;
        Hcalib.B[255] = 255;
    }


    void FullSystem::printResult(std::string file) {
        boost::unique_lock <boost::mutex> lock(trackMutex);
        boost::unique_lock <boost::mutex> crlock(shellPoseMutex);

        std::ofstream myfile;
        myfile.open(file.c_str());
        myfile << std::setprecision(15);

        for (FrameShell *s : allFrameHistory) {
            if (!s->poseValid) continue;

            if (setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;

            myfile << s->timestamp <<
                   " " << s->camToWorld.translation().transpose() <<
                   " " << s->camToWorld.so3().unit_quaternion().x() <<
                   " " << s->camToWorld.so3().unit_quaternion().y() <<
                   " " << s->camToWorld.so3().unit_quaternion().z() <<
                   " " << s->camToWorld.so3().unit_quaternion().w() << "\n";
        }
        myfile.close();
    }

    // push many tries
    // calculate residuals according to tries
    // get jacobian of residuals
    // optimize pose iteratively
    Vec4 FullSystem::trackNewCoarse(FrameHessian *fh) {

        assert(allFrameHistory.size() > 0);
        // set pose initialization.

        for (IOWrap::Output3DWrapper *ow : outputWrapper)
            ow->pushLiveFrame(fh);


        FrameHessian *lastF = coarseTracker->lastRef; // dump the last tracked host frame.

        AffLight aff_last_2_l = AffLight(0, 0); // init affine model a and b as 0
        // last frame to current frame hessian tries, vector of SE3 to store those proposed transformation matrix.
        // aligned allocator overdrive the memory allocate and release of SE3.
        std::vector<SE3, Eigen::aligned_allocator < SE3>>
        lastF_2_fh_tries;
        if (allFrameHistory.size() == 2) // this means it's in initialization.
            for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
                lastF_2_fh_tries.push_back(SE3()); // initialize with empty SE3
        else {
            FrameShell *slast = allFrameHistory[allFrameHistory.size() -
                                                2]; // -1 is the last one, -2 is the second last one
            FrameShell *sprelast = allFrameHistory[allFrameHistory.size() - 3]; // -3 is the third last one.
            // sprelast ... slast...
            SE3 slast_2_sprelast;
            SE3 lastF_2_slast;
            {    // lock on global pose consistency!
                boost::unique_lock <boost::mutex> crlock(shellPoseMutex);
                // this first convert last frame's points in camera coordinate into world coordinate
                // then convert from world coordinate into camera coordinate in sprelast frame,
                // now this point is in sprelast frame.
                slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld; // SE3 map last to prelast.
                lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld; // prelast to last.
                aff_last_2_l = slast->aff_g2l; // use the last affine model.
            }
            SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast. last -> prelast


            // get last delta-movement.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);    // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() *
                                       lastF_2_slast);    // assume double motion (frame skipped)
            lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() *
                                       lastF_2_slast); // assume half motion. this is equivalent as doing sqrt in matrix.
            lastF_2_fh_tries.push_back(
                    lastF_2_slast); // assume zero motion. this transform back to slast which is the second last frame in the frame history.
            lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.


            // just try a TON of different initializations (all rotations). In the end,
            // if they don't work they will only be tried on the coarsest level, which is super fast anyway.
            // also, if tracking rails here we loose, so we really, really want to avoid that.
            // here for loop only happens once. Don't know why they want to use this trick...
            for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta++) {
                // 25 different rotaitions. but they are all very small rotation change.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, 0, 0),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, rotDelta, 0),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, 0, rotDelta),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
            }

            if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) {
                lastF_2_fh_tries.clear();
                lastF_2_fh_tries.push_back(SE3());
            }
        }


        Vec3 flowVecs = Vec3(100, 100, 100); // updated from coarse tracker...
        SE3 lastF_2_fh = SE3(); // lastF_2_fh will be updated from lastF2_fh_tries, this will try to re-track multiple times
        AffLight aff_g2l = AffLight(0, 0);


        // as long as maxResForImmediateAccept is not reached, I'll continue through the options.
        // I'll keep track of the so-far best achieved residual for each level in achievedRes.
        // If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.


        Vec5 achievedRes = Vec5::Constant(NAN);
        bool haveOneGood = false;
        int tryIterations = 0;
        // try to track in all motions.
        for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) {
            AffLight aff_g2l_this = aff_last_2_l;
            SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
            // note that this is trackNestCoarse in CoarseTracker.cpp, not trackNewCoarse!!!
            // lastF_2_fh_this is the motions vector that was pushed tons of motions above.
            bool trackingIsGood = coarseTracker->trackNewestCoarse(
                    fh, lastF_2_fh_this, aff_g2l_this,
                    pyrLevelsUsed - 1,
                    achievedRes);    // in each level has to be at least as good as the last try.
            tryIterations++;

            if (i != 0) {
                printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
                       i,
                       i, pyrLevelsUsed - 1,
                       aff_g2l_this.a, aff_g2l_this.b,
                       achievedRes[0],
                       achievedRes[1],
                       achievedRes[2],
                       achievedRes[3],
                       achievedRes[4],
                       coarseTracker->lastResiduals[0],
                       coarseTracker->lastResiduals[1],
                       coarseTracker->lastResiduals[2],
                       coarseTracker->lastResiduals[3],
                       coarseTracker->lastResiduals[4]);
            }
            // achievedRes is updated in the trackNewestCoarse func

            // do we have a new winner?
            if (trackingIsGood && std::isfinite((float) coarseTracker->lastResiduals[0]) &&
                !(coarseTracker->lastResiduals[0] >= achievedRes[0])) {
                //printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
                flowVecs = coarseTracker->lastFlowIndicators;
                aff_g2l = aff_g2l_this;
                lastF_2_fh = lastF_2_fh_this;
                haveOneGood = true;
            }

            // take over achieved res (always).
            if (haveOneGood) {
                for (int i = 0; i < 5; i++) {
                    if (!std::isfinite((float) achievedRes[i]) || achievedRes[i] >
                                                                  coarseTracker->lastResiduals[i])    // take over if achievedRes is either bigger or NAN.
                        achievedRes[i] = coarseTracker->lastResiduals[i]; // use historical stable tracker
                }
            }


            if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)
                break;

        }

        if (!haveOneGood) {
            printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
            flowVecs = Vec3(0, 0, 0);
            aff_g2l = aff_last_2_l;
            lastF_2_fh = lastF_2_fh_tries[0];
        }

        lastCoarseRMSE = achievedRes; // coarse tracker residuals

        // no lock required, as fh is not used anywhere yet.
        // update the coarse tracked estimation into frame hessian shell for display and marginalization
        fh->shell->camToTrackingRef = lastF_2_fh.inverse();
        fh->shell->trackingRef = lastF->shell;
        fh->shell->aff_g2l = aff_g2l;
        fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;


        if (coarseTracker->firstCoarseRMSE < 0) // why this RMSE will below 0?
            coarseTracker->firstCoarseRMSE = achievedRes[0];

        if (!setting_debugout_runquiet)
            printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure,
                   achievedRes[0]);


        if (setting_logStuff) {
            (*coarseTrackingLog) << std::setprecision(16)
                                 << fh->shell->id << " "
                                 << fh->shell->timestamp << " "
                                 << fh->ab_exposure << " "
                                 << fh->shell->camToWorld.log().transpose() << " "
                                 << aff_g2l.a << " "
                                 << aff_g2l.b << " "
                                 << achievedRes[0] << " "
                                 << tryIterations << "\n";
        }

        // achievedRes is the converged residual, flowVecs is lastFlowIndicators from coarseTracker which is the converged optical flow.
        return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
    }

    // trace the immature point on the host frame on the target frame: FrameHessian* fh.
    // the key step is ph->traceOn() func, which perform epipolar search
    void FullSystem::traceNewCoarse(FrameHessian *fh) {
        // fh here is target frame.
        // lock the mapMutex on tracking thread.
        boost::unique_lock <boost::mutex> lock(mapMutex);

        int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

        Mat33f K = Mat33f::Identity();
        K(0, 0) = Hcalib.fxl();
        K(1, 1) = Hcalib.fyl();
        K(0, 2) = Hcalib.cxl();
        K(1, 2) = Hcalib.cyl();

        for (FrameHessian *host : frameHessians)        // go through all active frames
        {

            SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
            Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
            Vec3f Kt = K * hostToNew.translation().cast<float>();

            Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(),
                                                    fh->aff_g2l()).cast<float>();

            for (ImmaturePoint *ph : host->immaturePoints) {
                ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
                trace_total++;
            }
        }
//	printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
//			trace_total,
//			trace_good, 100*trace_good/(float)trace_total,
//			trace_skip, 100*trace_skip/(float)trace_total,
//			trace_badcondition, 100*trace_badcondition/(float)trace_total,
//			trace_oob, 100*trace_oob/(float)trace_total,
//			trace_out, 100*trace_out/(float)trace_total,
//			trace_uninitialized, 100*trace_uninitialized/(float)trace_total);
    }


    // this is the function directly optimize the toOptimize points and store the optimized one
    // into optimized.

    // this reductor is the main thread function, in this thread, they use the residuals of each frames
    // to optimize the immature points in range [min, max]
    void FullSystem::activatePointsMT_Reductor(
            std::vector<PointHessian *> *optimized,
            std::vector<ImmaturePoint *> *toOptimize,
            int min, int max, Vec10 *stats, int tid) {
        // min max is the index of toOptimize, Vec10* is Running*, tid is thread id.
        // ----------------------------------------------------------------
        // create temporal residual for the immature points
        // this temporal residual takes all the points in to optimize
        // that means every immature point to be optimized shares the same residual object
        // framehessians is the sliding window of frames. that explains this temproal residual
        // is set for each frame. every immature point in that frame share the same residual struct
        // and this struct describes the residual:

        ImmaturePointTemporaryResidual *tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
        for (int k = min; k < max; k++) {
            // in FullSystemOptPoint.cpp
            // optimizeImmaturePoint function
            (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr); // linearize residual, optimize point.
        }
        delete[] tr;
    }


// activate points on multi-thread.
    void FullSystem::activatePointsMT() {
        // this is all about setting parameters.
        if (ef->nPoints < setting_desiredPointDensity * 0.66)
            currentMinActDist -= 0.8;
        if (ef->nPoints < setting_desiredPointDensity * 0.8)
            currentMinActDist -= 0.5;
        else if (ef->nPoints < setting_desiredPointDensity * 0.9)
            currentMinActDist -= 0.2;
        else if (ef->nPoints < setting_desiredPointDensity)
            currentMinActDist -= 0.1;

        if (ef->nPoints > setting_desiredPointDensity * 1.5)
            currentMinActDist += 0.8;
        if (ef->nPoints > setting_desiredPointDensity * 1.3)
            currentMinActDist += 0.5;
        if (ef->nPoints > setting_desiredPointDensity * 1.15)
            currentMinActDist += 0.2;
        if (ef->nPoints > setting_desiredPointDensity)
            currentMinActDist += 0.1;

        if (currentMinActDist < 0) currentMinActDist = 0;
        if (currentMinActDist > 4) currentMinActDist = 4;

        if (!setting_debugout_runquiet)
            printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                   currentMinActDist, (int) (setting_desiredPointDensity), ef->nPoints);


        // remember frameHessians is the sliding window of frames.
        FrameHessian *newestHs = frameHessians.back();

        // make dist map.
        coarseDistanceMap->makeK(&Hcalib); // intrinsics in different scales.
        coarseDistanceMap->makeDistanceMap(frameHessians,
                                           newestHs); // record all points neigbours in bfsList1 and bfsList2

        //coarseTracker->debugPlotDistMap("distMap");
        // the immature points to optimize in threads. will be used in threadReduce
        std::vector<ImmaturePoint *> toOptimize;
        toOptimize.reserve(20000);


        // project all frames in the sliding window into last frame (newestHs)
        for (FrameHessian *host : frameHessians)        // go through all active frames
        {
            if (host == newestHs) continue; // newestHs is the last frameHessian
            SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
            // get Rotation and translation matrix.
            // will be applied in ptp.
            Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
            Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

            // loop all points in all frames in the sliding window...
            for (unsigned int i = 0; i < host->immaturePoints.size(); i += 1) {
                ImmaturePoint *ph = host->immaturePoints[i];
                ph->idxInImmaturePoints = i;

                // delete points that have never been traced successfully, or that are outlier on the last trace.
                if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER) {
//				immature_invalid_deleted++;
                    // remove point.
                    delete ph;
                    host->immaturePoints[i] = 0;
                    continue;
                }

                // can activate only if this is true.
                bool canActivate = (ph->lastTraceStatus == IPS_GOOD
                                    || ph->lastTraceStatus == IPS_SKIPPED
                                    || ph->lastTraceStatus == IPS_BADCONDITION
                                    || ph->lastTraceStatus == IPS_OOB)
                                   && ph->lastTracePixelInterval < 8
                                   && ph->quality > setting_minTraceQuality
                                   && (ph->idepth_max + ph->idepth_min) > 0;


                // if I cannot activate the point, skip it. Maybe also delete it.
                if (!canActivate) {
                    // if point will be out afterwards, delete it instead.
                    if (ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB) {
//					immature_notReady_deleted++;
                        delete ph;
                        host->immaturePoints[i] = 0;
                    }
//				immature_notReady_skipped++;
                    continue;
                }


                // see if we need to activate point due to distance map.
                Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) +
                            Kt * (0.5f * (ph->idepth_max + ph->idepth_min)); // point to point
                int u = ptp[0] / ptp[2] + 0.5f; // this u and v is the point projected to last frame hessian.
                int v = ptp[1] / ptp[2] + 0.5f;
                // if not OOB
                if ((u > 0 && v > 0 && u < wG[1] && v < hG[1])) {
                    // remember fwdWarpedIDDistFinal is updated in makeDistanceMap, which stores all the
                    // search distances (index k in the loop [0..40]).
                    // search distance plus the decimal part of ptp[0] which is x in world cordinate w.r.t last frame.
                    float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u + wG[1] * v] +
                                 (ptp[0] - floorf((float) (ptp[0])));
                    // holy shit, what this my_type is ?????
                    // why can't you find a proper name?????
                    // guessing: my type is a float number
                    // my_type should be a threshold term and this statement is to choose all points that not good enough,
                    // and throw them into the optimizer and re-estimate the depth.
                    if (dist >= currentMinActDist * ph->my_type) {
                        coarseDistanceMap->addIntoDistFinal(u, v);
                        toOptimize.push_back(ph);
                    }
                } else // OOB, just discard.
                {
                    delete ph;
                    host->immaturePoints[i] = 0;
                }
            }
        }


//	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
//			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

        // now toOptimize collected all points that requires to update the idepth estimation.
        // optimized is used to store those optimized points.
        std::vector<PointHessian *> optimized;
        optimized.resize(toOptimize.size());

        if (multiThreading) {
            // here boost:bind the activatePointsMT_Reductor with "&optimized, &toOptimize"
            // and left four arguments: _1, _2, _3, _4 corresponding to
            // "int, int, Running*, int" in
            // boost::function<void(int,int,Running*,int)> callPerIndex
            // and reduce function in threadReduce takes function pointer, first, end and step size as arguments
            // inline void reduce(boost::function<void(int,int,Running*,int)> callPerIndex, int first, int end, int stepSize = 0)
            // first is:
            // end is:
            // if stepsize is 0 then depends on first and end and num_threads

            // here this thread reducer is fed with toOptimize vector
            // and final results will push to optimized, notice they are both in reference
            // so since activePointsMT_Reductor takes 6 arguments in total, first two are passed here, last four will be
            // passed in function reduce() which contains the start and end position in toOptimize array, where
            // those points are throw into threads to optimize (linearize residual, GN iterative methods)
            treadReduce.reduce(
                    boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4),
                    0, toOptimize.size(), 50);
        } else {
            // run it in main tread.
            activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);
        }

        // loop all to optimize all points
        for (unsigned k = 0; k < toOptimize.size(); k++) {
            PointHessian *newpoint = optimized[k]; // dump optimized points, new optimized points
            ImmaturePoint *ph = toOptimize[k]; // this is the bad estimations

            if (newpoint != 0 && newpoint != (PointHessian * )((long) (-1))) {
                newpoint->host->immaturePoints[ph->idxInImmaturePoints] = 0; // cancel out the points that was estimated
                newpoint->host->pointHessians.push_back(newpoint);// record those optimized points as good point in host frame.
                ef->insertPoint(newpoint); // insert the converged active mature points into ef.
                for (PointFrameResidual *r : newpoint->residuals) // residuals in each points are created in makeKeyFrame func
                    ef->insertResidual(r); // residuals was pushed back to the energy function for optimization other immature points.
                assert(newpoint->efPoint != 0);
                delete ph; // since everything was took over by ef and host, toOptimize pointer is no longer needed.
            } else if (newpoint == (PointHessian * )((long) (-1)) || ph->lastTraceStatus == IPS_OOB) {
                // this kind of cast is so strange, why it's (long) (-1) -> this will give -1, what is (PointHessian*) -1?
                // cast -1 into PointHessian? => this gives 0xffffffffffffffff... which is the null pointer.........
                // okay, fair enough, 秀儿是你吗？
                delete ph;
                ph->host->immaturePoints[ph->idxInImmaturePoints] = 0;
            } else {
                assert(newpoint == 0 || newpoint == (PointHessian * )((long) (-1)));
            }
        }

        // loop all frames and align the immature points in memory (vector: immaturePoints).
        for (FrameHessian *host : frameHessians) {
            for (int i = 0; i < (int) host->immaturePoints.size(); i++) {
                if (host->immaturePoints[i] == 0) {
                    host->immaturePoints[i] = host->immaturePoints.back(); // move last entry back to fill the hole.
                    host->immaturePoints.pop_back();
                    i--;
                }
            }
        }


    }


    void FullSystem::activatePointsOldFirst() {
        assert(false);
    }
    // used in marginalization
    void FullSystem::flagPointsForRemoval() {
        assert(EFIndicesValid);

        std::vector<FrameHessian *> fhsToKeepPoints; // intuitive, no need to explain here.
        std::vector<FrameHessian *> fhsToMargPoints;

        //if(setting_margPointVisWindow>0)
        {
            // loop backwards, if frameHessian was not marked as marginalization, just keep that frame.
            // strange, i was initialized as fh->size() - 1, and i >= fh->size()?
            // this is not reachable, unless frameHessians changes when finished first statement.
            for (int i = ((int) frameHessians.size()) - 1; i >= 0 && i >= ((int) frameHessians.size()); i--)
                if (!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);
            // since the first loop is not as reliable, the author decide to loop again to remove those marginalized frame.
            for (int i = 0; i < (int) frameHessians.size(); i++)
                if (frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
        }



        //ef->setAdjointsF();
        //ef->setDeltaF(&Hcalib);
        int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

        for (FrameHessian *host : frameHessians)        // go through all active frames
        {
            for (unsigned int i = 0; i < host->pointHessians.size(); i++) {
                PointHessian *ph = host->pointHessians[i]; // loop all points in that frame
                if (ph == 0) continue; // skip empty points

                if (ph->idepth_scaled < 0 || ph->residuals.size() == 0) {
                    host->pointHessiansOut.push_back(ph); // typically negative idepth should be OOB points, marked as out.
                    ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                    host->pointHessians[i] = 0;
                    flag_nores++;
                } else if (ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization) {
                    // if point is marked as OOB or frame is going to marginalized, marginalization will happen on this point
                    flag_oob++;
                    if (ph->isInlierNew()) { // this will recycle some good points
                        flag_in++; // reclaim those good tracking points
                        int ngoodRes = 0;
                        for (PointFrameResidual *r : ph->residuals) { // reset these residual status
                            r->resetOOB(); // set the state to be IN, and state_NewState as OUTLIER
                            r->linearize(&Hcalib);
                            r->efResidual->isLinearized = false; // mark points to optimize later
                            r->applyRes(true);
                            if (r->efResidual->isActive()) {
                                r->efResidual->fixLinearizationF(ef);
                                ngoodRes++;
                            }
                        }
                        if (ph->idepth_hessian > setting_minIdepthH_marg) {
                            flag_inin++;
                            ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
                            host->pointHessiansMarginalized.push_back(ph);
                        } else {
                            ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                            host->pointHessiansOut.push_back(ph);
                        }


                    } else { // point is invalid for tracking
                        host->pointHessiansOut.push_back(ph);
                        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;


                        //printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
                    }

                    host->pointHessians[i] = 0;
                }
            }


            for (int i = 0; i < (int) host->pointHessians.size(); i++) {
                if (host->pointHessians[i] == 0) {
                    host->pointHessians[i] = host->pointHessians.back();
                    host->pointHessians.pop_back();
                    i--;
                }
            }
        }

    }


    void FullSystem::addActiveFrame(ImageAndExposure *image, int id) {

        if (isLost) return;
        boost::unique_lock <boost::mutex> lock(trackMutex);


        // =========================== add into allFrameHistory =========================
        FrameHessian *fh = new FrameHessian();
        FrameShell *shell = new FrameShell();
        shell->camToWorld = SE3();        // no lock required, as fh is not used anywhere yet.
        shell->aff_g2l = AffLight(0, 0);
        shell->marginalizedAt = shell->id = allFrameHistory.size();
        shell->timestamp = image->timestamp;
        shell->incoming_id = id;
        fh->shell = shell;
        allFrameHistory.push_back(shell);


        // =========================== make Images / derivatives etc. =========================
        fh->ab_exposure = image->exposure_time;
        fh->makeImages(image->image, &Hcalib);


        if (!initialized) {
            // use initializer!
            if (coarseInitializer->frameID < 0)    // first frame set. fh is kept by coarseInitializer.
            {
                // set first will set first frame in carseInitializer and select candidate points in that frame,
                // build average pyramid space... and many other preparation for trackFrame.
                coarseInitializer->setFirst(&Hcalib, fh);
            } else if (coarseInitializer->trackFrame(fh, outputWrapper))    // if SNAPPED
            {
                // convert all immature points into point hessians for active tracking, and initialize the depth prior
                // SE3 prior etc...
                initializeFromInitializer(fh); // successfully tracked, will go into initialization step.
                lock.unlock(); // unlock the mapMutex...
                // since needKF is true, fh will be set as keyframe.
                // frame will be projected to front UI and start all tracking thread
                deliverTrackedFrame(fh, true); // will make keyframe, select new points to track
            } else {
                // if still initializing
                fh->shell->poseValid = false;
                delete fh;
            }
            return;
        } else    // do front-end operation.
        {
            // for not initialized case, they are swapping the coarseTracker pointer
            // =========================== SWAP tracking reference?. =========================
            if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID) {
                boost::unique_lock <boost::mutex> crlock(coarseTrackerSwapMutex);
                CoarseTracker *tmp = coarseTracker;
                coarseTracker = coarseTracker_forNewKF;
                coarseTracker_forNewKF = tmp;
            }


            Vec4 tres = trackNewCoarse(fh);
            if (!std::isfinite((double) tres[0]) || !std::isfinite((double) tres[1]) ||
                !std::isfinite((double) tres[2]) || !std::isfinite((double) tres[3])) {
                printf("Initial Tracking failed: LOST!\n");
                isLost = true;
                return;
            }

            bool needToMakeKF = false;
            if (setting_keyframesPerSecond > 0) {
                needToMakeKF = allFrameHistory.size() == 1 ||
                               (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) >
                               0.95f / setting_keyframesPerSecond;
            } else {
                // this is when needToMakeFK <= 0
                // coarseTracker is now pointing to the last frame hessian
                Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
                                                           coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

                // BRIGHTNESS CHECK
                needToMakeKF = allFrameHistory.size() == 1 || //when there's only one frame
                               setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double) tres[1]) /
                               (wG[0] + hG[0]) +
                               setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double) tres[2]) /
                               (wG[0] + hG[0]) +
                               setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double) tres[3]) /
                               (wG[0] + hG[0]) +
                               setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float) refToFh[0])) > 1 ||
                               2 * coarseTracker->firstCoarseRMSE < tres[0];

            }


            for (IOWrap::Output3DWrapper *ow : outputWrapper)
                ow->publishCamPose(fh->shell, &Hcalib);


            lock.unlock();
            deliverTrackedFrame(fh, needToMakeKF);
            return;
        }
    }

    void FullSystem::deliverTrackedFrame(FrameHessian *fh, bool needKF) {


        if (linearizeOperation) { // linearized optimization
            if (goStepByStep && lastRefStopID != coarseTracker->refFrameID) { // this is debug stuff, frame by frame
                MinimalImageF3 img(wG[0], hG[0], fh->dI);
                IOWrap::displayImage("frameToTrack", &img); // deliver the frame to UI
                while (true) {
                    char k = IOWrap::waitKey(0);
                    if (k == ' ') break;
                    handleKey(k);
                }
                lastRefStopID = coarseTracker->refFrameID;
            } else handleKey(IOWrap::waitKey(1));

            // needKF was given as parameter on deliverTrackedFrame,
            if (needKF) makeKeyFrame(fh);
            else makeNonKeyFrame(fh);
        } else // no linearize operation
        {
            boost::unique_lock <boost::mutex> lock(trackMapSyncMutex);
            unmappedTrackedFrames.push_back(fh);
            if (needKF) needNewKFAfter = fh->shell->trackingRef->id;
            trackedFrameSignal.notify_all(); // start all tracking thread

            while (coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1) {
                mappedFrameSignal.wait(lock); // pause mapping thread to wait for tracking thread.
            }

            lock.unlock();
        }
    }

    void FullSystem::mappingLoop() {
        boost::unique_lock <boost::mutex> lock(trackMapSyncMutex);

        while (runMapping) {
            while (unmappedTrackedFrames.size() == 0) {
                trackedFrameSignal.wait(lock);
                if (!runMapping) return;
            }

            FrameHessian *fh = unmappedTrackedFrames.front();
            unmappedTrackedFrames.pop_front();


            // guaranteed to make a KF for the very first two tracked frames.
            if (allKeyFramesHistory.size() <= 2) {
                lock.unlock();
                makeKeyFrame(fh);
                lock.lock();
                mappedFrameSignal.notify_all();
                continue;
            }

            if (unmappedTrackedFrames.size() > 3)
                needToKetchupMapping = true;


            if (unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
            {
                lock.unlock();
                makeNonKeyFrame(fh);
                lock.lock();

                if (needToKetchupMapping && unmappedTrackedFrames.size() > 0) {
                    FrameHessian *fh = unmappedTrackedFrames.front();
                    unmappedTrackedFrames.pop_front();
                    {
                        boost::unique_lock <boost::mutex> crlock(shellPoseMutex);
                        assert(fh->shell->trackingRef != 0);
                        fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
                        fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
                    }
                    delete fh;
                }

            } else {
                if (setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id) {
                    lock.unlock();
                    makeKeyFrame(fh);
                    needToKetchupMapping = false;
                    lock.lock();
                } else {
                    lock.unlock();
                    makeNonKeyFrame(fh);
                    lock.lock();
                }
            }
            mappedFrameSignal.notify_all();
        }
        printf("MAPPING FINISHED!\n");
    }

    void FullSystem::blockUntilMappingIsFinished() {
        boost::unique_lock <boost::mutex> lock(trackMapSyncMutex);
        runMapping = false;
        trackedFrameSignal.notify_all();
        lock.unlock();

        mappingThread.join();

    }

    void FullSystem::makeNonKeyFrame(FrameHessian *fh) {
        // needs to be set by mapping thread. no lock required since we are in mapping thread.
        {
            boost::unique_lock <boost::mutex> crlock(shellPoseMutex);
            assert(fh->shell->trackingRef != 0);
            fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
            fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
        }

        traceNewCoarse(fh);
        delete fh;
    }

    void FullSystem::makeKeyFrame(FrameHessian *fh) {
        // needs to be set by mapping thread
        {
            boost::unique_lock <boost::mutex> crlock(shellPoseMutex);
            assert(fh->shell->trackingRef != 0);
            fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
            fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
        }

        traceNewCoarse(fh);

        boost::unique_lock <boost::mutex> lock(mapMutex);

        // =========================== Flag Frames to be Marginalized. =========================
        flagFramesForMarginalization(fh);


        // =========================== add New Frame to Hessian Struct. =========================
        fh->idx = frameHessians.size();
        frameHessians.push_back(fh);
        fh->frameID = allKeyFramesHistory.size();
        allKeyFramesHistory.push_back(fh->shell);
        ef->insertFrame(fh, &Hcalib);

        setPrecalcValues();



        // =========================== add new residuals for old points =========================
        int numFwdResAdde = 0;
        // but this loops all frames in the sliding window...
        for (FrameHessian *fh1 : frameHessians)        // go through all active frames
        {
            if (fh1 == fh) continue; // oh, I see, skip here... Im so blind...
            // now this will push all the (valid) residuals from the all the frame hessian projections into each mature point hessians
            for (PointHessian *ph : fh1->pointHessians) {
                PointFrameResidual *r = new PointFrameResidual(ph, fh1, fh);
                r->setState(ResState::IN);
                ph->residuals.push_back(r);
                ef->insertResidual(r);
                ph->lastResiduals[1] = ph->lastResiduals[0];
                ph->lastResiduals[0] = std::pair<PointFrameResidual *, ResState>(r, ResState::IN);
                numFwdResAdde += 1;
            }
        }




        // =========================== Activate Points (& flag for marginalization). =========================
        activatePointsMT();
        ef->makeIDX();




        // =========================== OPTIMIZE ALL =========================

        fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
        float rmse = optimize(setting_maxOptIterations);





        // =========================== Figure Out if INITIALIZATION FAILED =========================
        if (allKeyFramesHistory.size() <= 4) {
            if (allKeyFramesHistory.size() == 2 && rmse > 20 * benchmark_initializerSlackFactor) {
                printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
                initFailed = true;
            }
            if (allKeyFramesHistory.size() == 3 && rmse > 13 * benchmark_initializerSlackFactor) {
                printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
                initFailed = true;
            }
            if (allKeyFramesHistory.size() == 4 && rmse > 9 * benchmark_initializerSlackFactor) {
                printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
                initFailed = true;
            }
        }


        if (isLost) return;




        // =========================== REMOVE OUTLIER =========================
        removeOutliers();


        {
            boost::unique_lock <boost::mutex> crlock(coarseTrackerSwapMutex);
            coarseTracker_forNewKF->makeK(&Hcalib);
            coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);


            coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
            coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
        }


        debugPlot("post Optimize");






        // =========================== (Activate-)Marginalize Points =========================
        flagPointsForRemoval();
        ef->dropPointsF();
        getNullspaces(
                ef->lastNullspaces_pose,
                ef->lastNullspaces_scale,
                ef->lastNullspaces_affA,
                ef->lastNullspaces_affB);
        ef->marginalizePointsF();



        // =========================== add new Immature points & new residuals =========================
        makeNewTraces(fh, 0); // after dropping some points, we are going to select some new points in


        for (IOWrap::Output3DWrapper *ow : outputWrapper) {
            ow->publishGraph(ef->connectivityMap);
            ow->publishKeyframes(frameHessians, false, &Hcalib);
        }



        // =========================== Marginalize Frames =========================

        for (unsigned int i = 0; i < frameHessians.size(); i++)
            if (frameHessians[i]->flaggedForMarginalization) {
                marginalizeFrame(frameHessians[i]);
                i = 0;
            }


        printLogLine();
        //printEigenValLine();

    }

// note this function is invoked after the successful tracking.
    void FullSystem::initializeFromInitializer(FrameHessian *newFrame) {
        boost::unique_lock <boost::mutex> lock(mapMutex); // lock everything in this function.

        // add firstframe.
        FrameHessian *firstFrame = coarseInitializer->firstFrame; // let the coarseInitializer's host frame as fullsystem's host frame.
        firstFrame->idx = frameHessians.size(); // note that initialize may happen after reset. frameHessians record all tracked frames.
        frameHessians.push_back(firstFrame); // simple, just push the first frame into the frame window.
        firstFrame->frameID = allKeyFramesHistory.size(); // allKeyFramesHistory's back as id.
        allKeyFramesHistory.push_back(firstFrame->shell); // just pushback the transformation matrix to visualize
        // ########## need to explore the following two functions.
        ef->insertFrame(firstFrame, &Hcalib);
        setPrecalcValues();
        // #################################
        //int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
        //int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

        firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f); // reserve 20% of points in the first frame.
        firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f); // also reserve 20% of marginalized points
        firstFrame->pointHessiansOut.reserve(
                wG[0] * hG[0] * 0.2f); // the number of points that has been marginalize out.


        float sumID = 1e-5, numID = 1e-5;
        for (int i = 0;
             i < coarseInitializer->numPoints[0]; i++) // numPoints[0] is the point number in the largest scale lvl.
        {
            sumID += coarseInitializer->points[0][i].iR; // sumID is the sum of all inverse depth.
            numID++;
        }
        float rescaleFactor = 1 / (sumID / numID); // this rescale factor is the average depth in lvl 0. It normalized all iR to 1.

        // randomly sub-select the points I need.
        float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

        if (!setting_debugout_runquiet)
            printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100 * keepPercentage,
                   (int) (setting_desiredPointDensity), coarseInitializer->numPoints[0]);
        // to this step, initializer by default regard coarseInitializer has finished all scale space selected point initial tracking
        // and already get different point's valid depth estimation as prior.
        for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
            if (rand() / (float) RAND_MAX > keepPercentage) continue;
            // pointer locate in each selected point in lvl0
            Pnt *point = coarseInitializer->points[0] + i;
            // create a immature point out of the current point and current frame.
            ImmaturePoint *pt = new ImmaturePoint(point->u + 0.5f, point->v + 0.5f, firstFrame, point->my_type,
                                                  &Hcalib);
            // according to immature point class:
            // energyTH is calculated from the normalized color. so most of the color values in the boundaries will
            // be discard. Thus, the point will be deleted.
            if (!std::isfinite(pt->energyTH)) {
                delete pt;
                continue;
            }

            // !Notice: set all points idepths are 1. -> then update the idepth according to the next frame.
            // since this is initlization, idepth is the unknown variable, thus will set as 1 for temprory use.
            pt->idepth_max = pt->idepth_min = 1;
            // !Notice: so initially all points are set from immature point!
            // okay, now they use the immature point to create pointHessian...
            PointHessian *ph = new PointHessian(pt, &Hcalib); // point hessian should be the actively tracked map point.
            // delete the immature point ? no tracing? this point has uniformally depth: 1. where do you think you will
            // update them?
            delete pt; // delete the artifact: immaturePoint.
            if (!std::isfinite(ph->energyTH)) {
                delete ph;
                continue;
            }
            //idepthScaled is setting the normalized GroundTruth idepth
            ph->setIdepthScaled(point->iR * rescaleFactor); // initialize the idepth by iR
            //idepthZero is setting the estimated GroundTruth idepth, is used to calculate the error.
            ph->setIdepthZero(ph->idepth); // dump initial idepth
            ph->hasDepthPrior = true;
            ph->setPointStatus(
                    PointHessian::ACTIVE); // since it has valid depth prior, will be marked as active map point.

            firstFrame->pointHessians.push_back(ph); // now push the active point hessian to the host frame.
            ef->insertPoint(ph); // insert to energy function to optimize in the sliding window later on.
        }


        SE3 firstToNew = coarseInitializer->thisToNext; // coarse initializer has already got the SE3 matrix.
        firstToNew.translation() /= rescaleFactor; // rescale factor is normalized depth in lvl 0.


        // really no lock required, as we are initializing.
        {
            // but you still locked it, you can try to lock the local variables using lock_guard
            boost::unique_lock <boost::mutex> crlock(shellPoseMutex);
            // initialize the SE3 for the host frame hessian.
            firstFrame->shell->camToWorld = SE3();
            // initialize affine model.
            firstFrame->shell->aff_g2l = AffLight(0, 0);
            firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(), firstFrame->shell->aff_g2l);
            firstFrame->shell->trackingRef = 0;
            firstFrame->shell->camToTrackingRef = SE3();

            newFrame->shell->camToWorld = firstToNew.inverse();
            newFrame->shell->aff_g2l = AffLight(0, 0);
            newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(), newFrame->shell->aff_g2l);
            newFrame->shell->trackingRef = firstFrame->shell;
            newFrame->shell->camToTrackingRef = firstToNew.inverse(); // this firstToNew is from coarse initializer, set as SE3 prior.

        }

        initialized = true;
        printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int) firstFrame->pointHessians.size());
    }

    // okay, this selects the points frame latest frame.
    // I guess this author wanted to select those already mature points in the newframe, otherwise there's
    // no way to get gtDepth.
    // note that this is happening on each new frame come in.
    void FullSystem::makeNewTraces(FrameHessian *newFrame, float *gtDepth) {
        pixelSelector->allowFast = true;
        //int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
        // okay, this is same trick, select the desired number of points with locally higher gradients
        int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap, setting_desiredImmatureDensity);
        // reserve for those newly selected points, note that they might contains those already activated points
        newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
        //fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
        newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
        newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);

        // patternPadding here is 2. these two for loops are looping through all points inside the
        // padding to initialize as immature point.
        for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++)
            for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++) {
                int i = x + y * wG[0];
                if (selectionMap[i] == 0) continue;

                ImmaturePoint *impt = new ImmaturePoint(x, y, newFrame, selectionMap[i], &Hcalib);
                if (!std::isfinite(impt->energyTH)) delete impt;
                else newFrame->immaturePoints.push_back(impt);

            }
        //printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());

    }


    void FullSystem::setPrecalcValues() {
        for (FrameHessian *fh : frameHessians) {
            fh->targetPrecalc.resize(frameHessians.size());
            for (unsigned int i = 0; i < frameHessians.size(); i++)
                fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
        }

        ef->setDeltaF(&Hcalib);
    }


    void FullSystem::printLogLine() {
        if (frameHessians.size() == 0) return;

        if (!setting_debugout_runquiet)
            printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
                   allKeyFramesHistory.back()->id,
                   statistics_lastFineTrackRMSE,
                   ef->resInA,
                   ef->resInL,
                   ef->resInM,
                   (int) statistics_numForceDroppedResFwd,
                   (int) statistics_numForceDroppedResBwd,
                   allKeyFramesHistory.back()->aff_g2l.a,
                   allKeyFramesHistory.back()->aff_g2l.b,
                   frameHessians.back()->shell->id - frameHessians.front()->shell->id,
                   (int) frameHessians.size());


        if (!setting_logStuff) return;

        if (numsLog != 0) {
            (*numsLog) << allKeyFramesHistory.back()->id << " " <<
                       statistics_lastFineTrackRMSE << " " <<
                       (int) statistics_numCreatedPoints << " " <<
                       (int) statistics_numActivatedPoints << " " <<
                       (int) statistics_numDroppedPoints << " " <<
                       (int) statistics_lastNumOptIts << " " <<
                       ef->resInA << " " <<
                       ef->resInL << " " <<
                       ef->resInM << " " <<
                       statistics_numMargResFwd << " " <<
                       statistics_numMargResBwd << " " <<
                       statistics_numForceDroppedResFwd << " " <<
                       statistics_numForceDroppedResBwd << " " <<
                       frameHessians.back()->aff_g2l().a << " " <<
                       frameHessians.back()->aff_g2l().b << " " <<
                       frameHessians.back()->shell->id - frameHessians.front()->shell->id << " " <<
                       (int) frameHessians.size() << " " << "\n";
            numsLog->flush();
        }


    }


    void FullSystem::printEigenValLine() {
        if (!setting_logStuff) return;
        if (ef->lastHS.rows() < 12) return;


        MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
        MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
        int n = Hp.cols() / 8;
        assert(Hp.cols() % 8 == 0);

        // sub-select
        for (int i = 0; i < n; i++) {
            MatXX tmp6 = Hp.block(i * 8, 0, 6, n * 8);
            Hp.block(i * 6, 0, 6, n * 8) = tmp6;

            MatXX tmp2 = Ha.block(i * 8 + 6, 0, 2, n * 8);
            Ha.block(i * 2, 0, 2, n * 8) = tmp2;
        }
        for (int i = 0; i < n; i++) {
            MatXX tmp6 = Hp.block(0, i * 8, n * 8, 6);
            Hp.block(0, i * 6, n * 8, 6) = tmp6;

            MatXX tmp2 = Ha.block(0, i * 8 + 6, n * 8, 2);
            Ha.block(0, i * 2, n * 8, 2) = tmp2;
        }

        VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
        VecX eigenP = Hp.topLeftCorner(n * 6, n * 6).eigenvalues().real();
        VecX eigenA = Ha.topLeftCorner(n * 2, n * 2).eigenvalues().real();
        VecX diagonal = ef->lastHS.diagonal();

        std::sort(eigenvaluesAll.data(), eigenvaluesAll.data() + eigenvaluesAll.size());
        std::sort(eigenP.data(), eigenP.data() + eigenP.size());
        std::sort(eigenA.data(), eigenA.data() + eigenA.size());

        int nz = std::max(100, setting_maxFrames * 10);

        if (eigenAllLog != 0) {
            VecX ea = VecX::Zero(nz);
            ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
            (*eigenAllLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
            eigenAllLog->flush();
        }
        if (eigenALog != 0) {
            VecX ea = VecX::Zero(nz);
            ea.head(eigenA.size()) = eigenA;
            (*eigenALog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
            eigenALog->flush();
        }
        if (eigenPLog != 0) {
            VecX ea = VecX::Zero(nz);
            ea.head(eigenP.size()) = eigenP;
            (*eigenPLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
            eigenPLog->flush();
        }

        if (DiagonalLog != 0) {
            VecX ea = VecX::Zero(nz);
            ea.head(diagonal.size()) = diagonal;
            (*DiagonalLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
            DiagonalLog->flush();
        }

        if (variancesLog != 0) {
            VecX ea = VecX::Zero(nz);
            ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
            (*variancesLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
            variancesLog->flush();
        }

        std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
        (*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
        for (unsigned int i = 0; i < nsp.size(); i++)
            (*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " ";
        (*nullspacesLog) << "\n";
        nullspacesLog->flush();

    }

    void FullSystem::printFrameLifetimes() {
        if (!setting_logStuff) return;


        boost::unique_lock <boost::mutex> lock(trackMutex);

        std::ofstream *lg = new std::ofstream();
        lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
        lg->precision(15);

        for (FrameShell *s : allFrameHistory) {
            (*lg) << s->id
                  << " " << s->marginalizedAt
                  << " " << s->statistics_goodResOnThis
                  << " " << s->statistics_outlierResOnThis
                  << " " << s->movedByOpt;


            (*lg) << "\n";
        }


        lg->close();
        delete lg;

    }


    void FullSystem::printEvalLine() {
        return;
    }


}
