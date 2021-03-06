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


#pragma once

 
#include "util/NumType.h"
 
#include "FullSystem/HessianBlocks.h"
namespace dso
{


struct ImmaturePointTemporaryResidual
{
public:
	ResState state_state;
	double state_energy;
	ResState state_NewState;
	double state_NewEnergy;
	FrameHessian* target;
};


enum ImmaturePointStatus {
	IPS_GOOD=0,					// traced well and good
	IPS_OOB,					// OOB: end tracking & marginalize!
	IPS_OUTLIER,				// energy too high: if happens again: outlier!
	IPS_SKIPPED,				// traced well and good (but not actually traced).
	IPS_BADCONDITION,			// not traced because of bad condition.
	IPS_UNINITIALIZED};			// not even traced once.


class ImmaturePoint
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// static values
	float color[MAX_RES_PER_POINT]; // MAX_RES_PER_POINT is 8 here, why it's a 8 size float array?, because it stores 8 nearby points' intensity.
	// this weights and the pattern are exactly the SSD described in the paper. which
	// they calculate the residual pattern.
	float weights[MAX_RES_PER_POINT]; // again float array with length 8




	// gradient of Hessian? what is H?
	// answer: gradH is the gradient matrix aggregated from 8 different direction given in the residual pattern.
	Mat22f gradH; // gradient H, here H in the paper is noted as matrix. so this is gradient matrix.
	Vec2f gradH_ev;
	Mat22f gradH_eig;
	float energyTH; // 12*12*8 = 912, 12 is the energy threshold in setting, for each point, energy is the huber norm of residual.
	float u,v; // u and v are the coordinate in the image plane (host frame)
	FrameHessian* host; // yeah, this is the host frame that this immature point are located in and looking for reprojection candidates in the upcoming target frames.
	int idxInImmaturePoints; // this index helps frame to locate the immature points like this: host->immaturePoints[ph->idxInImmaturePoints]

	float quality; // quality capture the smallest energy and the second smallest energy, when the energy change is large (converge faster) quality is higher.

	float my_type; // this variable is for debugging... different type have different colors. (totally 4 types)

	float idepth_min; // as it says, min idepth, and max idepth. they are used to create the reprojection region for immature point to the target frame.
	float idepth_max;
	ImmaturePoint(int u_, int v_, FrameHessian* host_, float type, CalibHessian* HCalib); // constructor.
	~ImmaturePoint();
    // image alignment using Squared Sum differential (SSD)
	ImmaturePointStatus traceOn(FrameHessian* frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f &hostToFrame_affine, CalibHessian* HCalib, bool debugPrint=false);

	ImmaturePointStatus lastTraceStatus; // flag to see if OOB.
	Vec2f lastTraceUV; // never used.
	float lastTracePixelInterval; // idepth region area.

	float idepth_GT; // suppose to be the ground truth of idepth.

	// linearize residual calculation to simplify computation.
	double linearizeResidual(
			CalibHessian *  HCalib, const float outlierTHSlack,
			ImmaturePointTemporaryResidual* tmpRes,
			float &Hdd, float &bd,
			float idepth);
	float getdPixdd(
			CalibHessian *  HCalib,
			ImmaturePointTemporaryResidual* tmpRes,
			float idepth);

	float calcResidual(
			CalibHessian *  HCalib, const float outlierTHSlack,
			ImmaturePointTemporaryResidual* tmpRes,
			float idepth);

private:
};

}

