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
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "util/settings.h"

namespace dso
{

// is this part of linearization process for inverse depth?
EIGEN_STRONG_INLINE float derive_idepth(
		const Vec3f &t, const float &u, const float &v,
		const int &dx, const int &dy, const float &dxInterp,
		const float &dyInterp, const float &drescale)
{
    // du/dd = (t0 - t3*u)*(dt/dh), this equation means the partial of u w.r.t partial of inverse depth is equal to
    // (t0 - t3*u)*(dt/dh), dt(dxInterp) here is inverse depth of target frame, and dh(drescale) is hte inverse depth of host frame
    // remember the variable "dd" in CaorseInitializer function CalResAndGS, dd[idx] = dxInterp * dxdd + dyInterp * dydd
	return (dxInterp*drescale * (t[0]-t[2]*u)
			+ dyInterp*drescale * (t[1]-t[2]*v))*SCALE_IDEPTH;
}


// point reprojection, return false if it's OOB. normally used when there's precalculated KRKi and Kt matrix.
// invoked in ImmaturePoint::calcResidual function when calculating residuals.
EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt,const float &v_pt,
		const float &idepth,
		const Mat33f &KRKi, const Vec3f &Kt,
		float &Ku, float &Kv)
{
	Vec3f ptp = KRKi * Vec3f(u_pt,v_pt, 1) + Kt*idepth; // reprojection from [u_pt, v_pt] to [Ku, Kv]
	Ku = ptp[0] / ptp[2];
	Kv = ptp[1] / ptp[2];
	return Ku>1.1f && Kv>1.1f && Ku<wM3G && Kv<hM3G;
}



EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt,const float &v_pt,
		const float &idepth,
		const int &dx, const int &dy,
		CalibHessian* const &HCalib,
		const Mat33f &R, const Vec3f &t,
		float &drescale, float &u, float &v,
		float &Ku, float &Kv, Vec3f &KliP, float &new_idepth)
{
    // K^-1*P
	KliP = Vec3f(
			(u_pt+dx-HCalib->cxl())*HCalib->fxli(), // recover 3d x, fxli => 1/fx
			(v_pt+dy-HCalib->cyl())*HCalib->fyli(), // recover 3d y, fyli => 1/fy
			1);
    // extrinsic: T*K^-1*P -> point in new frame (3D)
    // why they multiply t by idepth ? why do it with inverse depth??????? shouldn't it be depth?
	Vec3f ptp = R * KliP + t*idepth; // R t is transformation matrix to new frame, so this ptp is 3d point in new frame
	drescale = 1.0f/ptp[2]; // depth in new frame, or you can say, "predicted depth", its actually = 1/t[2]*id = d/t[2]
	new_idepth = idepth*drescale; // new_idepth was never in use

	if(!(drescale>0)) return false;

	u = ptp[0] * drescale;
	v = ptp[1] * drescale;
	Ku = u*HCalib->fxl() + HCalib->cxl();
	Kv = v*HCalib->fyl() + HCalib->cyl();

	return Ku>1.1f && Kv>1.1f && Ku<wM3G && Kv<hM3G;
}




}

