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


 
#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

namespace dso
{


PointHessian::PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib)
{
	instanceCounter++;
	host = rawPoint->host;
	hasDepthPrior=false;

	idepth_hessian=0;
	maxRelBaseline=0;
	numGoodResiduals=0;

	// set static values & initialization.
	u = rawPoint->u;
	v = rawPoint->v;
	assert(std::isfinite(rawPoint->idepth_max));
	//idepth_init = rawPoint->idepth_GT;

	my_type = rawPoint->my_type;

	setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min)*0.5);
	setPointStatus(PointHessian::INACTIVE);

	int n = patternNum;
	memcpy(color, rawPoint->color, sizeof(float)*n);
	memcpy(weights, rawPoint->weights, sizeof(float)*n);
	energyTH = rawPoint->energyTH;

	efPoint=0;


}


void PointHessian::release()
{
	for(unsigned int i=0;i<residuals.size();i++) delete residuals[i];
	residuals.clear();
}


void FrameHessian::setStateZero(const Vec10 &state_zero)
{
	assert(state_zero.head<6>().squaredNorm() < 1e-20);

	this->state_zero = state_zero;


	for(int i=0;i<6;i++)
	{
		Vec6 eps; eps.setZero(); eps[i] = 1e-3;
		SE3 EepsP = Sophus::SE3::exp(eps);
		SE3 EepsM = Sophus::SE3::exp(-eps);
		SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
		SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
		nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);
	}
	//nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
	//nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

	// scale change
	SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_P_x0.translation() *= 1.00001;
	w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
	SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_M_x0.translation() /= 1.00001;
	w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
	nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);


	nullspaces_affine.setZero();
	nullspaces_affine.topLeftCorner<2,1>()  = Vec2(1,0);
	assert(ab_exposure > 0);
	nullspaces_affine.topRightCorner<2,1>() = Vec2(0, expf(aff_g2l_0().a)*ab_exposure);
};



void FrameHessian::release()
{
	// DELETE POINT
	// DELETE RESIDUAL
	for(unsigned int i=0;i<pointHessians.size();i++) delete pointHessians[i];
	for(unsigned int i=0;i<pointHessiansMarginalized.size();i++) delete pointHessiansMarginalized[i];
	for(unsigned int i=0;i<pointHessiansOut.size();i++) delete pointHessiansOut[i];
	for(unsigned int i=0;i<immaturePoints.size();i++) delete immaturePoints[i];


	pointHessians.clear();
	pointHessiansMarginalized.clear();
	pointHessiansOut.clear();
	immaturePoints.clear();
}

// this function make all images in different scale space
// three channels: image color, dx, dy
// and the gradient of image according to gamma response and down sampling local region.
void FrameHessian::makeImages(float* color, CalibHessian* HCalib)
{
    // loop all scale space to copy the
	for(int i=0;i<pyrLevelsUsed;i++)
	{
	    // this dIp allocate a new image for different scale.
	    // don't know yet why it's vector3f. maybe x, y, inverse depth.
	    // dI should be the graident of Image, but p, don't really konw. maybe projection?
	    // maybe d here is the depth, will figure out later on...
	    // !!!!! dI_l[idx][1] = dx;
        // !!!!! dI_l[idx][2] = dy;
        // now I understand that dx and dy are stored in the 2nd and 3rd channel.
        // and d here means gradient.
		dIp[i] = new Eigen::Vector3f[wG[i]*hG[i]];
		// abs squared gradient is storing all those absolute value of squared gradient of the image.
		// allocated the same size as dIp, but it's a float matrix, just one dimensional.
		absSquaredGrad[i] = new float[wG[i]*hG[i]];
	}
	// dI is the image on the largest scale.
	dI = dIp[0];


	// make d0
	// the largest scale of width and height
	int w=wG[0];
	int h=hG[0];
	// here assign the color image inside dI. this color only takes one dimension, grey scale?
    // note that dI is wG*hG*3. this dI[i][0] only assigned color to the first dimension.
	for(int i=0;i<w*h;i++)
		dI[i][0] = color[i];

    // this loop create the scale space by average pooling.
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
	    // here take different scales
		int wl = wG[lvl], hl = hG[lvl];
		// init the dI for each scale space. Since dIp has already initialized by new Vector3f...
		// here just point back to the original dIp.
		Eigen::Vector3f* dI_l = dIp[lvl];
        // pointer to the abs gradient layer.
		float* dabs_l = absSquaredGrad[lvl];
		// for the down-sampled images...
		if(lvl>0)
		{
		    // note that this is in the for loop that loop through all the scales.
		    // this lvlm1 is one layer above, which is larger image in the scale pyramid.
			int lvlm1 = lvl-1;
			// why they just take the wlm1 no hlm1?
			int wlm1 = wG[lvlm1];
			// dI_lm is obvious the larger image point to dIp, but why they are using Vector3f*
			// there's just one channel.
			Eigen::Vector3f* dI_lm = dIp[lvlm1];


            // oh, I see, this step is just for the down-sampling.
            // compress bottom layer's high quality to the higher layers, which has smaller scale.
			for(int y=0;y<hl;y++)
				for(int x=0;x<wl;x++)
				{
				    // for each point in the down sample layer
				    // why they choose 2*x, it's because their scale is power of 2 which implemented by
				    // bit-shift. 8>>1 = 4 ... something like that.
				    // now, I finally know that why they don't use the hlm1. because it's not necessary,
				    // they just average out the nearby 4 pixels on the upper bigger image into one
				    // this is literally equivilent to an average pooling with kernel of 2x2.
					dI_l[x + y*wl][0] = 0.25f * (dI_lm[2*x   + 2*y*wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1][0] +
												dI_lm[2*x   + 2*y*wlm1+wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1+wlm1][0]);
				}
		}
        // this is still in the scale space for loop
        // loop the image in current scale, just created by the for loop above.
		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
		    // horizental gradient of the image
			float dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx-1][0]);
			// vertical gradient of the image
			float dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx-wl][0]);

            // set all those illegal values to 0
			if(!std::isfinite(dx)) dx=0;
			if(!std::isfinite(dy)) dy=0;
            // so this dI_l Vector3f 2nd and 3rd channel is used here!!!!!
            // to store dx and dy.
			dI_l[idx][1] = dx;
			dI_l[idx][2] = dy;

            // dabs_l update the absgradient by the squared sum of gradient.
            // note that dabs_l is a pointer to a float image.
            // this is a non-linear gamma function of the gradient.
			dabs_l[idx] = dx*dx+dy*dy;
            // if use original intensity for pixel selection, and calibration hessian created.
            // to be honest, I still don't know why they are using hessian so often, will report here later.
			if(setting_gammaWeightsPixelSelect==1 && HCalib!=0)
			{
			    // setting_gammaWeightsPixelSelect can choose 0 which is use their defined weighted intensity choose
			    // if you know the gamma function you can try to set it to 1.
			    // -------------------------------------------------------------
			    // here getBGradOnly get the gradient of gamma function, which is 1 if dI_l[idx][0] is in [5..250] and decimal part is less than 0.5
			    // and 2 if dI_l[idx][0] decimal part is larger than 0.5, here dI_l[idx] is the pixel location, [0] is to index to grey color channel.
				float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
				// this makes gradient more sensitive to the decimal part of the color. but how can a color be some digit with .5 as decimal?
				// this doesn't make sense. all the image colors are suppose to be integers.
				// OH!!!! I know, when downsampling, it will create .5 part of the pixels! great job, this emphasize
				// the down-sampled part, which is rich in color pattern nearby 4 pixels....
				dabs_l[idx] *= gw*gw;	// convert to gradient of original color space (before removing response).
			}
		}
	}
}

void FrameFramePrecalc::set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib )
{
	this->host = host;
	this->target = target;

	SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
	PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
	PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();


    // calculate the warp matrix from target frame to the host grame.
	SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
	PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
	PRE_tTll = (leftToLeft.translation()).cast<float>();
	distanceLL = leftToLeft.translation().norm();


	Mat33f K = Mat33f::Zero();
	K(0,0) = HCalib->fxl();
	K(1,1) = HCalib->fyl();
	K(0,2) = HCalib->cxl();
	K(1,2) = HCalib->cyl();
	K(2,2) = 1;
	PRE_KRKiTll = K * PRE_RTll * K.inverse();
	PRE_RKiTll = PRE_RTll * K.inverse();
	PRE_KtTll = K * PRE_tTll;


	PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<float>();
	PRE_b0_mode = host->aff_g2l_0().b;
}

}

