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
#include "FullSystem/ImmaturePoint.h"
#include "math.h"

namespace dso
{
// this file is to optimize the point.

// TODO: find out what to optimize in the immature point?
// --------------------- explain: -------------------------
PointHessian* FullSystem::optimizeImmaturePoint(
		ImmaturePoint* point, int minObs,
		ImmaturePointTemporaryResidual* residuals)
{
    // this loops through each frame in the sliding window and
    // set up the point residual states
	int nres = 0;
	for(FrameHessian* fh : frameHessians)
	{
	    // in this loop residuals initialize all the states of the residual struct for point.
		if(fh != point->host) // add fh to the target frame of residuals
		{
			residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
			residuals[nres].state_NewState = ResState::OUTLIER;
			residuals[nres].state_state = ResState::IN;
			residuals[nres].target = fh;
			nres++; // nres is eventually the number of frame hessians that's not the current host frame.
		}
	}
	// see if this loops through all frames without expections
	assert(nres == ((int)frameHessians.size())-1);
    // print? and print used to be 50 prob? they must be a logging beast
	bool print = false;//rand()%50==0;
    // Those variables are initialized for the optmization for the idepth of the immature point
	float lastEnergy = 0; // initialize energy and H and b.
	/*
	 * ImmaturePoint::linearizeResidual function.
	 * Hdd += (hw*d_idepth)*d_idepth;
	 * bd += (hw*residual)*d_idepth;
	 * here the hw is the width and height production
	 * d_idepth is the derivative of inverse depth
	 * */
	float lastHdd=0; // notice this is the optimization for the point, which is only one depth to estimate. Hdd is Hassian of derivative of inverse depth
	float lastbd=0; // bd is b vector of inverse depth
	// currentIdepth just normalize with max and min idepth
	float currentIdepth=(point->idepth_max+point->idepth_min)*0.5f; // rough estimate idepth, average of the max and min estimation... give out the initial guess at least





	// loop through each frame in the sliding window
	// linearize residuals, residual state_NewState and state_NewEnergy will be updated in the linearization func.
	for(int i=0;i<nres;i++) // loop all frames in the sliding window
	{
	    // aggregate the linearized residual for the point in each frame.
	    // residuals is the pointer to the temporal residual, so residuals+i point to ith frame's residual
	    // here 1000 is the outlier threshold, if the energy is larger than that, considered as outlier.
	    // linearizeResidual do the following steps:
	    // 1. project the immature point's 8 residual pattern points into target frame, and accumulate the
	    //    reprojection error and normalized with huber norm and return as energyLeft
	    // 2. Hdd and bd are also accumulated from each d_idepth.
	    //    Hdd += (hw * d_idepth) * d_idepth;
        //    bd += (hw * residual) * d_idepth;
        // 3. Update the residuals.state_NewState according to the energyLeft.
		lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals+i,lastHdd, lastbd, currentIdepth); // acumulate energies sum_t{sum_p{E}}
		// set the state as outlier if exceed the energy threshold
		// otherwise, will still be ResState::IN.
		residuals[i].state_state = residuals[i].state_NewState; // this state_NewState was already updated in linearizeResidual
		// set the energy as 0
		residuals[i].state_energy = residuals[i].state_NewEnergy; // state_NewEnergy is now energyLeft.
		// by doing this we wipe out the residual state in the temporal residual struct
		// and kept all the residuals aggregated into lastEnergy
	}
    // if there's computation error for the energy or last
	if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act) // Singularity, or failed to converge
	{
		if(print)
			printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
				nres, lastHdd, lastEnergy);
		return 0;
	}

	if(print) printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n" ,
			nres, lastHdd,lastEnergy,currentIdepth);

	float lambda = 0.1; // gradient move step size, lambda is the trust region...
    // -------------------------gauss-newton ---------------------------------------------
    // Hdx = b
    // dx = H^-1b
    // x = x + dx
    // H', b', id', e = LinearizeResidual(x), x here is the idepth...
    // H <- H'
    // b <- b'
    // loop...
	for(int iteration=0;iteration<setting_GNItsOnPointActivation;iteration++)
	{
		float H = lastHdd;
		H *= 1+lambda; // H = J^TJ(1 + lambda)! this is LM method ...
		float step = (1.0/H) * lastbd; // step is the gradient of update delta: dx = H^-1b
		float newIdepth = currentIdepth - step; // use step to update newIdepth estimation: x = x + dx

		float newHdd=0; float newbd=0; float newEnergy=0; // clear out all Hdd, bd and Energy, estimate Residual again:
		// collect the energy on the updated idepth, to see if energy decreases
		for(int i=0;i<nres;i++) // loop the whole sliding window once more, update the Hdd, bd, and idepth, and residuals
			newEnergy += point->linearizeResidual(&Hcalib, 1, residuals+i,newHdd, newbd, newIdepth);

		if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)
		{
			if(print) printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
					nres,
					newHdd,
					lastEnergy);
			return 0;
		}

		if(print) printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
				(true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
				iteration,
				log10(lambda),
				"",
				lastEnergy, newEnergy, newIdepth);

		if(newEnergy < lastEnergy) // if energy error decreases, we accept this step and update the estimations.
		{
		    // update lastHdd and lastbd ... according to the new ones.
			currentIdepth = newIdepth;
			lastHdd = newHdd;
			lastbd = newbd;
			lastEnergy = newEnergy;
			for(int i=0;i<nres;i++)
			{
				residuals[i].state_state = residuals[i].state_NewState; // update the residual states accordingly
				residuals[i].state_energy = residuals[i].state_NewEnergy;
			}

			lambda *= 0.5; // decrease the gradient stepsize (to be more careful and fine-grind), lambda here is the trust region
		}
		else
		{
			lambda *= 5; // else enlarge the step size to stride further, this can easily create over shoot and not converge...
		}

		if(fabsf(step) < 0.0001*currentIdepth) // converged.
			break;
	}
	// if for loop can reach successfully here, it for sure converged. or didnt get to the loop at all.
    // so currentIdepth should be the converged depth estimation with GN method.
	if(!std::isfinite(currentIdepth))
	{
		printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems. 秀儿，是你么？
	}


	int numGoodRes=0;
	// see if the point is still observable in the other frames.
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN) numGoodRes++;

	if(numGoodRes < minObs) // again, point doesn't have enough observations, bad point.
	{
		if(print) printf("OptPoint: OUTLIER!\n");
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}



	PointHessian* p = new PointHessian(point, &Hcalib);
	// check if point has valid energyTH. energyTH was maintained on constructor of ImmaturePoint
	if(!std::isfinite(p->energyTH)) {delete p; return (PointHessian*)((long)(-1));}

	p->lastResiduals[0].first = 0;
	p->lastResiduals[0].second = ResState::OOB;
	p->lastResiduals[1].first = 0;
	p->lastResiduals[1].second = ResState::OOB;
	p->setIdepthZero(currentIdepth); // use the new optimized estimation of idepth as it's idepth estimation
	p->setIdepth(currentIdepth);
	p->setPointStatus(PointHessian::ACTIVE); // active map point.
    // update all residuals
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN) // update those map points
		{
			PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target);
			r->state_NewEnergy = r->state_energy = 0; // clear out those residual values for optimization since it's optimized.
			r->state_NewState = ResState::OUTLIER;
			r->setState(ResState::IN); // set the state as IN which means this point is map point now.
			p->residuals.push_back(r); // push the residual describer above back to point.

			if(r->target == frameHessians.back())
			{
				p->lastResiduals[0].first = r; // last Residuals records residuals for last two frames.
				p->lastResiduals[0].second = ResState::IN;
			}
			else if(r->target == (frameHessians.size()<2 ? 0 : frameHessians[frameHessians.size()-2])) // if size is 2, just first one, else, the second last one.
			{
				p->lastResiduals[1].first = r; // simple, dump the residual pointer
				p->lastResiduals[1].second = ResState::IN; // and set the state as IN. which is map point accepted residual.
			}
		}

	if(print) printf("point activated!\n");

	statistics_numActivatedPoints++;
	return p; // p is the optimized point now.
}



}
