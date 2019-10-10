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


#include "FullSystem/PixelSelector2.h"

// 



#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalFuncs.h"

namespace dso {


    PixelSelector::PixelSelector(int w, int h) {
        randomPattern = new unsigned char[w * h];
        std::srand(3141592);    // want to be deterministic.
        for (int i = 0; i < w * h; i++) randomPattern[i] = rand() & 0xFF; // & 0xFF will keep last 8 bits, and set 0 of rest. which is make a random number in range [0..255]

        currentPotential = 3; // this current potiential search field is initialized as 3 which is a relative large space. will shrink on MakeMaps function to select enough points.


        gradHist = new int[100 * (1 + w / 32) * (1 + h / 32)];
        ths = new float[(w / 32) * (h / 32) + 100];
        thsSmoothed = new float[(w / 32) * (h / 32) + 100];

        allowFast = false;
        gradHistFrame = 0;
    }

    PixelSelector::~PixelSelector() {
        delete[] randomPattern;
        delete[] gradHist;
        delete[] ths;
        delete[] thsSmoothed;
    }

    int computeHistQuantil(int *hist, float below) {
        // hist[0] is 1024
        int th = hist[0] * below + 0.5f;
        // why loop 90 times? I know this is for finding something below the threshold, but why it's 90?
        // hist is 32*32 which is 1024 size list, why just loop first 90 entities?
        for (int i = 0; i < 90; i++) {
            // now I understand, this way they want to locate all those histogram that under threshold, and return the index
            // if the index is above 90, they will just return 90 instead.
            // ################################
            // this operation keeps minus first 90 elements in the hist. and when it reach to 0, just return that index.
            th -= hist[i + 1]; // start from i+1 because hist[0] has been used to define th.
            // return i!!!!!! this is the index! how can you return the index?
            if (th < 0) return i;
        }
        return 90;
    }

    // holy shit, this is exactly HOG.... absSquaredGrad is the Magnitude of gradient.....
    // they apply the HOG on a sliding window with kernel size 32*32, which means they are
    // applying HOG on the smallest scale space......
    void PixelSelector::makeHists(const FrameHessian *const fh) {
        gradHistFrame = fh;
        // mapmax0 get the sum squared gradient of the framehessian.
        // absSquaredGrad[0] is the original level gradient, first layer is the largest layer.
        float *mapmax0 = fh->absSquaredGrad[0]; // remember absSquaredGrad is the gradient of the image applied gamma function

        int w = wG[0];
        int h = hG[0];

        int w32 = w / 32; // note that 32 is 2^5 which means w32 and h32 is the smallest scale in this program.
        int h32 = h / 32;
        thsStep = w32;

        for (int y = 0; y < h32; y++)
            for (int x = 0; x < w32; x++) {
                // the stride is 32, so for each x and y, they down sample original gradient map to w*h/32*32 size.
                float *map0 = mapmax0 + 32 * x + 32 * y * w;
                int *hist0 = gradHist;// + 50*(x+y*w32);
                // set the first 50 values as 0? why set as 0? why it's 50?
                // guess: 50 in the front, 50 in the back, why all all those padding? and why the padding is 100?
                memset(hist0, 0, sizeof(int) * 50);
                // for each step in 32 by 32 square:
                // I think this is a kernel. 32*32 which collect the gradient and sum into a histogram (size of 50)
                for (int j = 0; j < 32; j++)
                    for (int i = 0; i < 32; i++) {
                        // it should be the cordinate in the original image frame, same as jt
                        // if they found out of bound, continue
                        // note that this is it and jt, I think this just prevent those points
                        // that is oob on the bigger scale where they fails to project into the current scale space.
                        int it = i + 32 * x;
                        int jt = j + 32 * y;
                        if (it > w - 2 || jt > h - 2 || it < 1 || jt < 1) continue;
                        // g turns out to be the index of hist0, and hist0 point to gradHist, and hist0 was padded 50 in front
                        // map0[i+j*w] is the gradient in that 32*32 square. consider it loopes to h32 and w32, this step
                        // might want to get the square by square histogram in that image. what those histograms to do with
                        // navigation?
                        // ####### understood. ##########
                        // this g is the statistic results for the histogram.
                        int g = sqrtf(map0[i + j * w]);
                        // remmeber that hist0 is 0 for first 50 and gradHist is at least 100 size.
                        if (g > 48) g = 48;
                        // this is to collect different graident level, and set the level [0-49] totally 50 different gradients.
                        hist0[g + 1]++;
                        // why will they add this hist0[0]? so hist0 will always be 50.
                        hist0[0]++;
                    }
                // find the threshold that the histogram below the setting threshold. return the found index before 90
                // ths[x, y] threshold in the downsampled image (size: [h32, w32]).
                // setting_minGradHistCut is 0.5
                // setting_minGradHistAdd is 7
                // this ths record each point in the w32 h32 scale, with their HOG threshold index plus an offset.
                ths[x + y * w32] = computeHistQuantil(hist0, setting_minGradHistCut) + setting_minGradHistAdd;
            }
        // loop the downsampled threshold image to smooth the thresholds,
        // remember that ths records the threshold indices!
        // they statistically collect the frequency of gradients, and map into histogram and filter out those small
        // thresholds' indices, and now they are smoothing the indices!!! what the hell, why this indices are smoothed>!#!#!
        for (int y = 0; y < h32; y++)
            for (int x = 0; x < w32; x++) {
                // add the up down left right threshold indice numbers and average with different directions added numbers
                // this controls the edges,
                // for example:
                // if they right to the right most, they will only calculate the up and down, then devide by 2
                float sum = 0, num = 0;
                // add sum when x > 0 add again when x < w32 - 1, and ...
                // overall, they added sum 9 times. if x in [1...w32-1] y in [1...h32-1]
                if (x > 0) {
                    // this add sum an extra term to include the top most row.
                    if (y > 0) {
                        num++;
                        sum += ths[x - 1 + (y - 1) * w32];
                    }
                    // this add sum the threshold on the bottom row.
                    if (y < h32 - 1) {
                        num++;
                        sum += ths[x - 1 + (y + 1) * w32];
                    }
                    // this reaches no matter what y is, so it add up 3 times if in [1...h32-1]
                    num++;
                    sum += ths[x - 1 + (y) * w32];
                }

                if (x < w32 - 1) {
                    if (y > 0) {
                        num++;
                        sum += ths[x + 1 + (y - 1) * w32];
                    }
                    if (y < h32 - 1) {
                        num++;
                        sum += ths[x + 1 + (y + 1) * w32];
                    }
                    num++;
                    sum += ths[x + 1 + (y) * w32];
                }

                if (y > 0) {
                    num++;
                    sum += ths[x + (y - 1) * w32];
                }
                if (y < h32 - 1) {
                    num++;
                    sum += ths[x + (y + 1) * w32];
                }
                num++;
                sum += ths[x + y * w32];
                // smoothed threshold indices?>>!@#@!# why you go to smooth those indices?>>>!@#
                // now I understand: they keep track on how many times they aggregated sum,
                // and do a final normalization according to num.
                // ################################################
                // so, here it should be a normalized threshold index
                // for each pixel location in the smallest scale space.
                thsSmoothed[x + y * w32] = (sum / num) * (sum / num); // mean normalized neighbourhood.

            }


    }

    // like this name, make the selectionMap
    // don't know yet what the selection map works for.
    // will update here later.
    // ---------------update----------------------
    // selection map marks the point should be selected and mark them with different scale space index.
    // notice the recursion inside, it will recursively search shirnked search potential area. (currentPotential = 3 --> shrink to 1... if there's not enough points.)
    // here this currentPotential or pot or any other potential is regard as stride as in convolution.
    int PixelSelector::makeMaps(
            const FrameHessian *const fh,
            float *map_out, float density, int recursionsLeft, bool plot, float thFactor) {
        // here the float* map_out is the status map in the coarse initializer. what they are working with is to update
        // the status map to feed back to initializer.
        // ####################################
        // *map_out point to the selectionMap.
        float numHave = 0;
        float numWant = density;
        float quotia;
        int idealPotential = currentPotential; // I think this current potential should be a percentage... but why it's int?
        // answer: idealPotential is a converged potential space which should be int, and represent the search range

//	if(setting_pixelSelectionUseFast>0 && allowFast)
//	{
//		memset(map_out, 0, sizeof(float)*wG[0]*hG[0]);
//		std::vector<cv::KeyPoint> pts;
//		cv::Mat img8u(hG[0],wG[0],CV_8U);
//		for(int i=0;i<wG[0]*hG[0];i++)
//		{
//			float v = fh->dI[i][0]*0.8;
//			img8u.at<uchar>(i) = (!std::isfinite(v) || v>255) ? 255 : v;
//		}
//		cv::FAST(img8u, pts, setting_pixelSelectionUseFast, true);
//		for(unsigned int i=0;i<pts.size();i++)
//		{
//			int x = pts[i].pt.x+0.5;
//			int y = pts[i].pt.y+0.5;
//			map_out[x+y*wG[0]]=1;
//			numHave++;
//		}
//
//		printf("FAST selection: got %f / %f!\n", numHave, numWant);
//		quotia = numWant / numHave;
//	}
//	else
        {




            // the number of selected pixels behaves approximately as
            // K / (pot+1)^2, where K is a scene-dependent constant.
            // we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.

            if (fh != gradHistFrame) makeHists(fh);

            // select!
            Eigen::Vector3i n = this->select(fh, map_out, currentPotential, thFactor);

            // sub-select!
            numHave = n[0] + n[1] + n[2]; // see, this n is n2 n3 and n4, the total point selected.
            quotia = numWant / numHave; // quotia is obvious, no need to explain...

            // by default we want to over-sample by 40% just to be sure.
            // if K = 1.4 numHave... that means currentPotential is ~0.2 (1.2*1.2 ~ 1.44)
            float K = numHave * (currentPotential + 1) * (currentPotential + 1);
            idealPotential = sqrtf(K / numWant) - 1;    // round down.
            if (idealPotential < 1) idealPotential = 1;

            if (recursionsLeft > 0 && quotia > 1.25 && currentPotential > 1) {
                //re-sample to get more points!
                // potential needs to be smaller
                if (idealPotential >= currentPotential)
                    idealPotential = currentPotential - 1;

                //		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
                //				100*numHave/(float)(wG[0]*hG[0]),
                //				100*numWant/(float)(wG[0]*hG[0]),
                //				currentPotential,
                //				idealPotential);
                currentPotential = idealPotential;
                return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
            } else if (recursionsLeft > 0 && quotia < 0.25) {
                // re-sample to get less points!

                if (idealPotential <= currentPotential)
                    idealPotential = currentPotential + 1;

                //		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
                //				100*numHave/(float)(wG[0]*hG[0]),
                //				100*numWant/(float)(wG[0]*hG[0]),
                //				currentPotential,
                //				idealPotential);
                currentPotential = idealPotential;
                return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);

            }
        }

        int numHaveSub = numHave;
        if (quotia < 0.95) {
            int wh = wG[0] * hG[0];
            int rn = 0;
            unsigned char charTH = 255 * quotia;
            for (int i = 0; i < wh; i++) {
                if (map_out[i] != 0) {
                    if (randomPattern[rn] > charTH) {
                        map_out[i] = 0;
                        numHaveSub--;
                    }
                    rn++;
                }
            }
        }

//	printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d -> %d. Subsampled to %.2f%%\n",
//			100*numHave/(float)(wG[0]*hG[0]),
//			100*numWant/(float)(wG[0]*hG[0]),
//			currentPotential,
//			idealPotential,
//			100*numHaveSub/(float)(wG[0]*hG[0]));
        currentPotential = idealPotential;


        if (plot) {
            int w = wG[0];
            int h = hG[0];


            MinimalImageB3 img(w, h);

            for (int i = 0; i < w * h; i++) {
                float c = fh->dI[i][0] * 0.7;
                if (c > 255) c = 255;
                img.at(i) = Vec3b(c, c, c);
            }
            IOWrap::displayImage("Selector Image", &img);

            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    int i = x + y * w;
                    if (map_out[i] == 1)
                        img.setPixelCirc(x, y, Vec3b(0, 255, 0));
                    else if (map_out[i] == 2)
                        img.setPixelCirc(x, y, Vec3b(255, 0, 0));
                    else if (map_out[i] == 4)
                        img.setPixelCirc(x, y, Vec3b(0, 0, 255));
                }
            IOWrap::displayImage("Selector Pixels", &img);
        }

        return numHaveSub;
    }

    // this function loops through different scale space and for each space, they select the point with
    // dx and dy collinear with the random direction.
    Eigen::Vector3i PixelSelector::select(const FrameHessian *const fh,
                                          float *map_out, int pot, float thFactor) {
        // int pot is potential, defined in the class, but what is potential defined in paper?
        // my guess is pot here is a scale factor for how much search space they want to cover
        // in each scale space.
        // the map0 now is used to define image.... notation abuse, understood.
        Eigen::Vector3f const *const map0 = fh->dI;
        // 3 levels of sum suared gradients. 0 is the largest and 2 is the smallest
        float *mapmax0 = fh->absSquaredGrad[0];
        float *mapmax1 = fh->absSquaredGrad[1];
        float *mapmax2 = fh->absSquaredGrad[2];

        // why they don't define h1, h2???? do they default set the image as squared?????
        // because they reshaped the matrix into 1d, so only width is needed
        int w = wG[0];
        int w1 = wG[1];
        int w2 = wG[2];
        // only define hte first layer of height?
        int h = hG[0];


        const Vec2f directions[16] = {
                Vec2f(0, 1.0000),
                Vec2f(0.3827, 0.9239),
                Vec2f(0.1951, 0.9808),
                Vec2f(0.9239, 0.3827),
                Vec2f(0.7071, 0.7071),
                Vec2f(0.3827, -0.9239),
                Vec2f(0.8315, 0.5556),
                Vec2f(0.8315, -0.5556),
                Vec2f(0.5556, -0.8315),
                Vec2f(0.9808, 0.1951),
                Vec2f(0.9239, -0.3827),
                Vec2f(0.7071, -0.7071),
                Vec2f(0.5556, 0.8315),
                Vec2f(0.9808, -0.1951),
                Vec2f(1.0000, 0.0000),
                Vec2f(0.1951, -0.9808)};

        // 4 different status, that makes the map_out 0 to 4 mapping of status pixel by pixel.
        // this is allocation of status write for map_out and pass back to the coarseInitializer
        // for now, I still don't know what those status used for, but will fill up later on
        //TODO: find out the function of PixelSelectorStatus
        // ##################
        // update: here there's 4 channels of status, which make the map_out or SelectionMap
        // a w*h*4 matrix.
        memset(map_out, 0, w * h * sizeof(PixelSelectorStatus));


        // down weighting constant, don't know what they exactly doing.
        float dw1 = setting_gradDownweightPerLevel;
        float dw2 = dw1 * dw1;
        //------------------------------------Explain of loop:----------------------------------------------

        // this loop is searching in different scales.

        // first scale y4, x4 which is looping trough the whole image space

        // second scale y3, x3 which looping through the 4*pot size small patch start at y4, x4.

        // now on y2, x2 and y1, x1 should be even smaller size of patch to search.

        // this is equivalent to the convolution through different scale space.

        // and they change directions in each loop step in each scale space.

        // eventually, in the loop body, they capture those largest dirNorm (which means gradient vector are collinear

        // with the random direction)

        // intuitively, we can regard all point like that are matched point and considered as selected point.

        //----------------------------------------------------------------------------------------
        // n3 n2 and n4 are index of vector returned. increased at each for loop.
        int n3 = 0, n2 = 0, n4 = 0;
        // pot here is the potential defined in the function or passed from coarse initializer
        // here I understand as the stride.
        // !Notice: this starts from y4 and x4 loop.
        for (int y4 = 0; y4 < h; y4 += (4 * pot))
            for (int x4 = 0; x4 < w; x4 += (4 * pot)) {
                // 4*pot, why it's 4*pot? h-y4?
                int my3 = std::min((4 * pot), h - y4);
                // w-x4? so select the min value of 4*pot or w-x4, this x index will always be 4*pot when it reach to the end of
                // the matrix, and why they will only get the last few steps? here, 4*pot under setting, is 12, so they will
                // always be 12 until x4 reaches end of w. but x4 will jump each 4*pot for each iteration.
                int mx3 = std::min((4 * pot), w - x4);
                int bestIdx4 = -1;
                float bestVal4 = 0;
                // 0xF is 15, and randomPattern[n2] is randomPattern[0] for now, so this is to sample the same direction.
                // shouldn't this be n4? randomPattern[n2] ?
                // randomPattern is generated on the constructor.
                Vec2f dir4 = directions[randomPattern[n2] & 0xF];
                // !Notice: this is y3 and x3 loop.
                for (int y3 = 0; y3 < my3; y3 += (2 * pot))
                    for (int x3 = 0; x3 < mx3; x3 += (2 * pot)) {
                        int x34 = x3 + x4;
                        int y34 = y3 + y4;
                        int my2 = std::min((2 * pot), h - y34);
                        int mx2 = std::min((2 * pot), w - x34);
                        int bestIdx3 = -1;
                        float bestVal3 = 0;
                        // shouldn't this be n3?
                        Vec2f dir3 = directions[randomPattern[n2] & 0xF];
                        for (int y2 = 0; y2 < my2; y2 += pot)
                            for (int x2 = 0; x2 < mx2; x2 += pot) {
                                int x234 = x2 + x34;
                                int y234 = y2 + y34;
                                int my1 = std::min(pot, h - y234);
                                int mx1 = std::min(pot, w - x234);
                                int bestIdx2 = -1;
                                // so this controls the loop for the update on n2..
                                // n2 is the key looper to choose direction, so that means if
                                // you didn't find any gradient that has higher direction norm
                                float bestVal2 = 0;
                                // this should be n2.
                                Vec2f dir2 = directions[randomPattern[n2] & 0xF];
                                // seems like they are searching around different scales, now comes to the 1 stride
                                for (int y1 = 0; y1 < my1; y1 += 1)
                                    for (int x1 = 0; x1 < mx1; x1 += 1) {
                                        assert(x1 + x234 < w);
                                        assert(y1 + y234 < h);
                                        // loop the small patch in different big strides, now I understand they are searching in different
                                        // patches are small when they reach to the end of the strides.
                                        int idx = x1 + x234 + w * (y1 + y234); // x234 = x2 + x3 + x4... same as y234, this is just offsets.
                                        int xf = x1 + x234;
                                        int yf = y1 + y234;

                                        if (xf < 4 || xf >= w - 5 || yf < 4 || yf > h - 4) continue;

                                        // this is the pixels index, why the hell they will down weight those index??????
                                        // beacause xf>>5 = xf/2^5 = xf/32, which is the smallest scale
                                        // this is indexing thsSmoothed[x32, y32]
                                        float pixelTH0 = thsSmoothed[(xf >> 5) + (yf >> 5) * thsStep]; // thsStep is w32
                                        // down weight those index will shrink those points in the threshold.
                                        // multiply dw1 and dw1 they lifted the bar of the threshold.
                                        float pixelTH1 = pixelTH0 * dw1;
                                        float pixelTH2 = pixelTH1 * dw2;

                                        // this is the single abs gradient in the local index.
                                        // thFactor now is 2...
                                        float ag0 = mapmax0[idx];
                                        // this pixelTH0 is just the min threshold for the gradient
                                        // in order to find out more valid gradient, they use histogram to store all
                                        // the gradients, and normalize throughout the down sampled image
                                        // now they just want to use this threshold to pick up point according to gradient
                                        // now I understand why they name it as pixel selector...
                                        if (ag0 > pixelTH0 * thFactor) {
                                            // this will give the last two scales of abs gradient
                                            // remember this map0 is the dI, dI is 3 channels, color, dx, dy,
                                            // now tail<2> selects dx and dy channel.
                                            // this explained why they use ag0d -> d means gradient.
                                            // ag absolute gradient? 0 represent scale 0 which is the original image.
                                            Vec2f ag0d = map0[idx].tail<2>();
                                            // ag0d.dot(dir2) will give the direction norm? dot product will be
                                            // zero if ag0d is perpendicular to dir2...
                                            // dir2 is the random direction sampled by n2...
                                            // that means n2 will change if dir2 is not perpendicular to ag0d.
                                            // which means they are finding all the non-rthonormal basis...
                                            // and eventually converge to the minimal angle and until they are
                                            // in the same direction...
                                            float dirNorm = fabsf((float) (ag0d.dot(dir2)));
                                            if (!setting_selectDirectionDistribution) dirNorm = ag0;
                                            // bestIdx2,3,4 are used to update the n2 n3 and n4
                                            // if it's not orthogonal, then update bestVal2...
                                            // and if it's less angle, the higher the dirNorm
                                            // this step is to align the direction to gradient angle...
                                            if (dirNorm > bestVal2) {
                                                bestVal2 = dirNorm;
                                                bestIdx2 = idx;
                                                bestIdx3 = -2;
                                                bestIdx4 = -2;
                                            }
                                        }
                                        // this continue will jump the loop
                                        // because they found the best alignment on this loop,
                                        // move to another x1...
                                        if (bestIdx3 == -2) continue;
                                        // same as above, this is for the second scale space. which
                                        // is w/2, h/2 size gradient image.
                                        // again, will find the most aligned direction along image gradient.
                                        // here why on mapmax1 they shrink size of x and y
                                        // because mapmax1 itself is the smaller sized map. width height is w/2 h/2
                                        // + 0.25 is to compensate for what? still unknown, will investigate later...
                                        float ag1 = mapmax1[(int) (xf * 0.5f + 0.25f) + (int) (yf * 0.5f + 0.25f) * w1];
                                        if (ag1 > pixelTH1 * thFactor) {
                                            Vec2f ag0d = map0[idx].tail<2>();
                                            float dirNorm = fabsf((float) (ag0d.dot(dir3)));
                                            if (!setting_selectDirectionDistribution) dirNorm = ag1;

                                            if (dirNorm > bestVal3) {
                                                bestVal3 = dirNorm;
                                                bestIdx3 = idx;
                                                bestIdx4 = -2;
                                            }
                                        }
                                        if (bestIdx4 == -2) continue;

                                        float ag2 = mapmax2[(int) (xf * 0.25f + 0.125) +
                                                            (int) (yf * 0.25f + 0.125) * w2];
                                        if (ag2 > pixelTH2 * thFactor) {
                                            Vec2f ag0d = map0[idx].tail<2>();
                                            float dirNorm = fabsf((float) (ag0d.dot(dir4)));
                                            if (!setting_selectDirectionDistribution) dirNorm = ag2;

                                            if (dirNorm > bestVal4) {
                                                bestVal4 = dirNorm;
                                                bestIdx4 = idx;
                                            }
                                        }
                                    }
                                // from this if all those following code are recording the map_out with those matched points
                                // and marked in different scales.
                                // map_out[some_index] = 1 means that this is the match point searched by the smallest scale.
                                if (bestIdx2 > 0) {
                                    map_out[bestIdx2] = 1;
                                    bestVal3 = 1e10;
                                    n2++; // this way it increased!!!
                                }
                            }

                        if (bestIdx3 > 0) {
                            map_out[bestIdx3] = 2; // mark the selected point in scale 2, which is even larger scale
                            bestVal4 = 1e10;
                            n3++;
                        }
                    }

                if (bestIdx4 > 0) {
                    map_out[bestIdx4] = 4; // this is the largest scale, which literally covers the whole image piexls.
                    n4++;
                }
            }
        // after finished the loop above, all the point that are suppose to be a match was selected and marked in map_out.
        // here map_out has 3 different scales. but every point marked in map_out are selected.
        // n3 n2 and n4 are the point size selected in different scales.
        return Eigen::Vector3i(n2, n3, n4);
    }


}

