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
#include "util/settings.h"
#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "fstream"

namespace dso
{



// reads interpolated element from a uchar* array
// SSE2 optimization possible
EIGEN_ALWAYS_INLINE float getInterpolatedElement(const float* const mat, const float x, const float y, const int width)
{
	//stats.num_pixelInterpolations++;

	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const float* bp = mat +ix+iy*width;


	float res =   dxdy * bp[1+width]
				+ (dy-dxdy) * bp[width]
				+ (dx-dxdy) * bp[1]
				+ (1-dx-dy+dxdy) * bp[0];

	return res;
}

EIGEN_ALWAYS_INLINE Eigen::Vector3f getInterpolatedElement43(const Eigen::Vector4f* const mat, const float x, const float y, const int width)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const Eigen::Vector4f* bp = mat +ix+iy*width;


	return dxdy * *(const Eigen::Vector3f*)(bp+1+width)
	        + (dy-dxdy) * *(const Eigen::Vector3f*)(bp+width)
	        + (dx-dxdy) * *(const Eigen::Vector3f*)(bp+1)
			+ (1-dx-dy+dxdy) * *(const Eigen::Vector3f*)(bp);
}

// get the normalized 3 channel (normalized from the decimal part of x and y)
EIGEN_ALWAYS_INLINE Eigen::Vector3f getInterpolatedElement33(const Eigen::Vector3f* const mat, const float x, const float y, const int width)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix; // x and y are float, so dx dy are the float part
	float dy = y - iy;
	float dxdy = dx*dy; // what is this for? why multiply dx and dy?
	const Eigen::Vector3f* bp = mat +ix+iy*width; // this is the pointer point to the 3 channels in [ix, iy] location in dI or other scales.

    /**
     *      x x  -> 1-dx-dy+dx*dy dx-dx*dy
     *      x x  ->     dy-dx*dy    dx*dy
     *
     *      This means that the original channels will get some part of the right and below
     *      information as normalization. This could be regard as linear interpolation.
     */
	return dxdy * *(const Eigen::Vector3f*)(bp+1+width) // point on one point right one row below
	        + (dy-dxdy) * *(const Eigen::Vector3f*)(bp+width) // point on the row below
	        + (dx-dxdy) * *(const Eigen::Vector3f*)(bp+1) // point on the right
			+ (1-dx-dy+dxdy) * *(const Eigen::Vector3f*)(bp); // original point.
}

EIGEN_ALWAYS_INLINE Eigen::Vector3f getInterpolatedElement33OverAnd(const Eigen::Vector3f* const mat, const bool* overMat, const float x, const float y, const int width, bool& over_out)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const Eigen::Vector3f* bp = mat +ix+iy*width;

	const bool* bbp = overMat +ix+iy*width;
	over_out = bbp[1+width] && bbp[1] && bbp[width] && bbp[0];

	return dxdy * *(const Eigen::Vector3f*)(bp+1+width)
	        + (dy-dxdy) * *(const Eigen::Vector3f*)(bp+width)
	        + (dx-dxdy) * *(const Eigen::Vector3f*)(bp+1)
			+ (1-dx-dy+dxdy) * *(const Eigen::Vector3f*)(bp);
}
EIGEN_ALWAYS_INLINE Eigen::Vector3f getInterpolatedElement33OverOr(const Eigen::Vector3f* const mat, const bool* overMat, const float x, const float y, const int width, bool& over_out)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const Eigen::Vector3f* bp = mat +ix+iy*width;

	const bool* bbp = overMat +ix+iy*width;
	over_out = bbp[1+width] || bbp[1] || bbp[width] || bbp[0];

	return dxdy * *(const Eigen::Vector3f*)(bp+1+width)
	        + (dy-dxdy) * *(const Eigen::Vector3f*)(bp+width)
	        + (dx-dxdy) * *(const Eigen::Vector3f*)(bp+1)
			+ (1-dx-dy+dxdy) * *(const Eigen::Vector3f*)(bp);
}

// just return the interpolated color channel so that it's 31, interpolated33 means get intensity, Ix, Iy.
EIGEN_ALWAYS_INLINE float getInterpolatedElement31(const Eigen::Vector3f* const mat, const float x, const float y, const int width)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const Eigen::Vector3f* bp = mat +ix+iy*width; // the original coordinate of the element


	return dxdy * (*(const Eigen::Vector3f*)(bp+1+width))[0] // dxdy is the production of decimal part of x and y, bp+1+width is the bottom right pixel coordinate
	        + (dy-dxdy) * (*(const Eigen::Vector3f*)(bp+width))[0] // one point below
	        + (dx-dxdy) * (*(const Eigen::Vector3f*)(bp+1))[0] // one point right
			+ (1-dx-dy+dxdy) * (*(const Eigen::Vector3f*)(bp))[0]; // original point
}

EIGEN_ALWAYS_INLINE Eigen::Vector3f getInterpolatedElement13BiLin(const float* const mat, const float x, const float y, const int width)
{
	int ix = (int)x;
	int iy = (int)y;
	const float* bp = mat +ix+iy*width;

	float tl = *(bp);
	float tr = *(bp+1);
	float bl = *(bp+width);
	float br = *(bp+width+1);

	float dx = x - ix;
	float dy = y - iy;
	float topInt = dx * tr + (1-dx) * tl;
	float botInt = dx * br + (1-dx) * bl;
	float leftInt = dy * bl + (1-dy) * tl;
	float rightInt = dy * br + (1-dy) * tr;

	return Eigen::Vector3f(
			dx * rightInt + (1-dx) * leftInt,
			rightInt-leftInt,
			botInt-topInt);
}

// interpolate the 3 channel image pixel values (color, dx, dy) according to the bi-direction linear interpolation.
// return a linearly interpolated color, dx, dy vector corresponding to the offered x y cordinate in image plane.
EIGEN_ALWAYS_INLINE Eigen::Vector3f getInterpolatedElement33BiLin(const Eigen::Vector3f* const mat, const float x, const float y, const int width)
{
    // here mat is dI, which is the trace for fine tracking.
	int ix = (int)x; // x = u+dx, dx is the dx in static pattern 8
	int iy = (int)y; // y = v+dy, dy is the dy in static pattern 8
	// bp is the point in the offset position.
	const Eigen::Vector3f* bp = mat +ix+iy*width; // bp is the base point, which is the [ix, iy] pixel position in dI.
    // bp is still a vector3f store [x, y, z]
    // tl is x,y
    // notice unlike
	float tl = (*(bp))[0]; //tl = top left
	// tr is x+1, y
	float tr = (*(bp+1))[0]; // tr = top right
	// bl is x, y+1
	float bl = (*(bp+width))[0]; // bl = bottom left
	// lr is x+1, y+1
	float br = (*(bp+width+1))[0]; // br = bottom right

	float dx = x - ix; // float part of the x coordinate
	float dy = y - iy; // float part of the y coordinate
	float topInt = dx * tr + (1-dx) * tl;
	float botInt = dx * br + (1-dx) * bl;
	float leftInt = dy * bl + (1-dy) * tl;
	float rightInt = dy * br + (1-dy) * tr;

	return Eigen::Vector3f(
			dx * rightInt + (1-dx) * leftInt, // so they do an weighted shift on this term. depending on the decimal value
			rightInt-leftInt, // rightInt - leftInt = (1-dy)*(tr-tl) -> tr-tl is dx on color space.
			botInt-topInt); // botInt - topInt = (1-dx)*(bl-tl) -> bl - tl is dy on the color space.
}
EIGEN_ALWAYS_INLINE float getInterpolatedElement11Cub(const float* const p, const float x)	// for x=0, this returns p[1].
{
	return p[1] + 0.5f * x*(p[2] - p[0] + x*(2.0f*p[0] - 5.0f*p[1] + 4.0f*p[2] - p[3] + x*(3.0f*(p[1] - p[2]) + p[3] - p[0])));
}

EIGEN_ALWAYS_INLINE Eigen::Vector2f getInterpolatedElement12Cub(const float* const p, const float x)	// for x=0, this returns p[1].
{
	float c1 = 0.5f * (p[2] - p[0]);
	float c2 = p[0]-2.5f*p[1]+2*p[2]-0.5f*p[3];
	float c3 = 0.5f*(3.0f*(p[1]-p[2])+p[3]-p[0]);
	float xx = x*x;
	float xxx = xx*x;
	return Eigen::Vector2f(p[1] + x*c1 + xx*c2 + xxx*c3, c1 + x*2.0f*c2 + xx*3.0f*c3);
}
EIGEN_ALWAYS_INLINE Eigen::Vector2f getInterpolatedElement32Cub(const Eigen::Vector3f* const p, const float x)	// for x=0, this returns p[1].
{
	float c1 = 0.5f * (p[2][0] - p[0][0]);
	float c2 = p[0][0]-2.5f*p[1][0]+2*p[2][0]-0.5f*p[3][0];
	float c3 = 0.5f*(3.0f*(p[1][0]-p[2][0])+p[3][0]-p[0][0]);
	float xx = x*x;
	float xxx = xx*x;
	return Eigen::Vector2f(p[1][0] + x*c1 + xx*c2 + xxx*c3, c1 + x*2.0f*c2 + xx*3.0f*c3);
}

EIGEN_ALWAYS_INLINE float getInterpolatedElement11BiCub(const float* const mat, const float x, const float y, const int width)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	const float* bp = mat +ix+iy*width;

	float val[4];
	val[0] =getInterpolatedElement11Cub(bp-width-1, dx);
	val[1] =getInterpolatedElement11Cub(bp-1, dx);
	val[2] =getInterpolatedElement11Cub(bp+width-1, dx);
	val[3] =getInterpolatedElement11Cub(bp+2*width-1, dx);

	float dy = y - iy;
	return getInterpolatedElement11Cub(val, dy);
}
EIGEN_ALWAYS_INLINE Eigen::Vector3f getInterpolatedElement13BiCub(const float* const mat, const float x, const float y, const int width)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	const float* bp = mat +ix+iy*width;

	float val[4];
	float grad[4];
	Eigen::Vector2f v = getInterpolatedElement12Cub(bp-width-1, dx);
	val[0] = v[0]; grad[0] = v[1];

	v = getInterpolatedElement12Cub(bp-1, dx);
	val[1] = v[0]; grad[1] = v[1];

	v = getInterpolatedElement12Cub(bp+width-1, dx);
	val[2] = v[0]; grad[2] = v[1];

	v = getInterpolatedElement12Cub(bp+2*width-1, dx);
	val[3] = v[0]; grad[3] = v[1];

	float dy = y - iy;
	v = getInterpolatedElement12Cub(val, dy);

	return Eigen::Vector3f(v[0], getInterpolatedElement11Cub(grad, dy), v[1]);
}

EIGEN_ALWAYS_INLINE Eigen::Vector3f getInterpolatedElement33BiCub(const Eigen::Vector3f* const mat, const float x, const float y, const int width)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	const Eigen::Vector3f* bp = mat +ix+iy*width;

	float val[4];
	float grad[4];
	Eigen::Vector2f v = getInterpolatedElement32Cub(bp-width-1, dx);
	val[0] = v[0]; grad[0] = v[1];

	v = getInterpolatedElement32Cub(bp-1, dx);
	val[1] = v[0]; grad[1] = v[1];

	v = getInterpolatedElement32Cub(bp+width-1, dx);
	val[2] = v[0]; grad[2] = v[1];

	v = getInterpolatedElement32Cub(bp+2*width-1, dx);
	val[3] = v[0]; grad[3] = v[1];

	float dy = y - iy;
	v = getInterpolatedElement12Cub(val, dy);

	return Eigen::Vector3f(v[0], getInterpolatedElement11Cub(grad, dy), v[1]);
}

EIGEN_ALWAYS_INLINE Eigen::Vector4f getInterpolatedElement44(const Eigen::Vector4f* const mat, const float x, const float y, const int width)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const Eigen::Vector4f* bp = mat +ix+iy*width;


	return dxdy * *(bp+1+width)
	        + (dy-dxdy) * *(bp+width)
	        + (dx-dxdy) * *(bp+1)
			+ (1-dx-dy+dxdy) * *(bp);
}

EIGEN_ALWAYS_INLINE Eigen::Vector2f getInterpolatedElement42(const Eigen::Vector4f* const mat, const float x, const float y, const int width)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const Eigen::Vector4f* bp = mat +ix+iy*width;


	return dxdy * *(const Eigen::Vector2f*)(bp+1+width)
	        + (dy-dxdy) * *(const Eigen::Vector2f*)(bp+width)
	        + (dx-dxdy) * *(const Eigen::Vector2f*)(bp+1)
			+ (1-dx-dy+dxdy) * *(const Eigen::Vector2f*)(bp);
}



inline Vec3f makeRainbowf3F(float id)
{
	id *= freeDebugParam3;
	if(id < 0)
		return Vec3f(1,1,1);

	int icP = id;
	float ifP = id-icP;
	icP = icP%3;

	if(icP == 0) return Vec3f((1-ifP), ifP,     0);
	if(icP == 1) return Vec3f(0,           (1-ifP), ifP);
	if(icP == 2) return Vec3f(ifP,     0,           (1-ifP));
	assert(false);
	return Vec3f(1,1,1);
}

inline Vec3b makeRainbow3B(float id)
{
	id *= freeDebugParam3;
	if(!(id > 0))
		return Vec3b(255,255,255); // white

	int icP = id;
	float ifP = id-icP;
	icP = icP%3;

	if(icP == 0) return Vec3b(255*(1-ifP), 255*ifP,     0);
	if(icP == 1) return Vec3b(0,           255*(1-ifP), 255*ifP);
	if(icP == 2) return Vec3b(255*ifP,     0,           255*(1-ifP));
	return Vec3b(255,255,255);
}

inline Vec3b makeJet3B(float id)
{
	if(id <= 0) return Vec3b(128,0,0);
	if(id >= 1) return Vec3b(0,0,128);

	int icP = (id*8);
	float ifP = (id*8)-icP;

	if(icP == 0) return Vec3b(255*(0.5+0.5*ifP), 		    		  0,     					0);
	if(icP == 1) return Vec3b(255, 					  255*(0.5*ifP),     					0);
	if(icP == 2) return Vec3b(255, 				  255*(0.5+0.5*ifP),     					0);
	if(icP == 3) return Vec3b(255*(1-0.5*ifP), 					255,     					255*(0.5*ifP));
	if(icP == 4) return Vec3b(255*(0.5-0.5*ifP), 					255,     					255*(0.5+0.5*ifP));
	if(icP == 5) return Vec3b(0, 						255*(1-0.5*ifP),     					255);
	if(icP == 6) return Vec3b(0, 						255*(0.5-0.5*ifP),     					255);
	if(icP == 7) return Vec3b(0, 					  				  0,     					255*(1-0.5*ifP));
	return Vec3b(255,255,255);
}

inline Vec3b makeRedGreen3B(float val)	// 0 = red, 1=green, 0.5=yellow.
{
	if(val < 0) return Vec3b(0,0,255);
	else if(val < 0.5) return Vec3b(0,255*2*val,255);
	else if(val < 1) return Vec3b(0,255,255-255*2*(val-0.5));
	else return Vec3b(0,255,0);

}





}
