/// Copyright (c) 2012 by Gao Han, Mark Matten, Xin Sun.
/// Copy freely.
#pragma once
#ifndef ANDRES_VISION_HESSIAN_MATRIX_HXX
#define ANDRES_VISION_HESSIAN_MATRIX_HXX

#include <cmath>

#include "marray.hxx"
#include "marray_hdf5.hxx"
#include "convolution.hxx"

namespace vision {

template<class T>
class HessianMatrixEigenvalues{
public:
	typedef T Value;

	HessianMatrixEigenvalues();
	HessianMatrixEigenvalues(const T, const size_t);

	Value& sigma();
	Value sigma() const;
	size_t& radius();
	size_t radius() const;
	size_t margin() const;	
	size_t numberOfValues(const size_t) const;
	void compute(const marray::View<Value>&, marray::Marray<Value>&) const;

private:
	Value sigma_;
	size_t radius_;
};

template<class T>
inline
HessianMatrixEigenvalues<T>::HessianMatrixEigenvalues() 
:	sigma_(static_cast<Value>(1)),
	radius_(4)
{}

template<class T>
inline
HessianMatrixEigenvalues<T>::HessianMatrixEigenvalues(
	const Value sigma, 
	const size_t radius
) 
:	sigma_(sigma),
	radius_(radius)
{}

template<class T>
inline typename HessianMatrixEigenvalues<T>::Value& 
HessianMatrixEigenvalues<T>::sigma() {
	return sigma_;
}

template<class T>
inline typename HessianMatrixEigenvalues<T>::Value 
HessianMatrixEigenvalues<T>::sigma() const {
	return sigma_;
}

template<class T>
inline size_t& 
HessianMatrixEigenvalues<T>::radius() {
	return radius_;
}

template<class T>
inline size_t 
HessianMatrixEigenvalues<T>::radius() const {
	return radius_;
}

template<class T>
inline size_t 
HessianMatrixEigenvalues<T>::margin() const {
	return radius_;
}	

template<class T>
inline size_t 
HessianMatrixEigenvalues<T>::numberOfValues(
	const size_t dimension
) const {
	if (dimension == 2) {
		return 2;
	}
	else if (dimension == 3) {
		return 3;
	}
	else {
		std::runtime_error("Improper dimension.");
	}
}

template<class T>
void 
HessianMatrixEigenvalues<T>::compute(
	const marray::View<Value>& in, 
	marray::Marray<Value>& out
) const {
	if (in.dimension() == 2) {
		size_t shape[] = {in.shape(0), in.shape(1), 2};
		out.resize(marray::SkipInitialization, shape, shape + 3);
		// create derivative mask
		marray::Marray<T> derivativeMask;
		vision::derivativeOfGaussian<T>(sigma_, radius_, derivativeMask);
		// create second derivative mask
		marray::Marray<T> secondDerivativeMask;
		vision::secondDerivativeOfGaussian<T>(sigma_, radius_, secondDerivativeMask);

		marray::Marray<T> dxx;
		marray::Marray<T> dyy;
		vision::secondDerive<T>(sigma_, radius_, 0, in, dxx);
		vision::secondDerive<T>(sigma_, radius_, 1, in, dyy);
		marray::Marray<T> dx;
		vision::derive<T>(sigma_, radius_, 0, in, dx);

		marray::Marray<T> dxy;
		vision::derive<T>(sigma_, radius_, 1, dx, dxy);
		hessianMatrixEigenvalues<T>(dxx, dxy, dyy, out);  
	}
	else if (in.dimension() == 3) {
		size_t shape[] = {in.shape(0), in.shape(1), in.shape(2), 3};
		out.resize(marray::SkipInitialization, shape, shape + 4);
		// create derivative mask
		marray::Marray<T> derivativeMask;
		vision::derivativeOfGaussian<T>(sigma_, radius_, derivativeMask);
		// create second derivative mask
		marray::Marray<T> secondDerivativeMask;
		vision::secondDerivativeOfGaussian<T>(sigma_, radius_, secondDerivativeMask);

		marray::Marray<T> dxx;
		marray::Marray<T> dyy;
		marray::Marray<T> dzz;
		vision::secondDerive<T>(sigma_, radius_, 0, in, dxx);
		vision::secondDerive<T>(sigma_, radius_, 1, in, dyy);
		vision::secondDerive<T>(sigma_, radius_, 2, in, dzz);
		marray::Marray<T> dx;
		marray::Marray<T> dy;
		vision::derive<T>(sigma_, radius_, 0, in, dx);
		vision::derive<T>(sigma_, radius_, 1, in, dy);

		marray::Marray<T> dxy;
		marray::Marray<T> dxz;
		marray::Marray<T> dyz;
		vision::derive<T>(sigma_, radius_, 1, dx, dxy);
		vision::derive<T>(sigma_, radius_, 2, dx, dxz);
		vision::derive<T>(sigma_, radius_, 2, dy, dyz);   
		hessianMatrixEigenvalues<T>(dxx, dyy, dzz, dxy, dxz, dyz, out);  
	}
	else {
		std::runtime_error("Improper dimension for image.");
	}
}

template<class T>
inline void 
hessianMatrixEigenvalues(
	const marray::Marray<T>& dxx, 
	const marray::Marray<T>& dxy, 
	const marray::Marray<T>& dyy,
	marray::Marray<T> & out
) {
	size_t shapeOut[] = {dxx.shape(0), dyy.shape(1), 2};
	out.resize(marray::SkipInitialization, shapeOut, shapeOut + 3);

	for (size_t y = 0; y < dyy.shape(1); ++y) 
	for (size_t x = 0; x < dxx.shape(0); ++x) {
		const T a = dxx(x, y);
		const T b = dxy(x, y);
		const T c = dxy(x, y);
		const T d = dyy(x, y);
		const T B = -(a + d);
		const T C = a * d - b * b;
		vision::roots(1, B, C, out(x, y, 0), out(x, y, 1));
	}
}

template<class T>
inline void 
hessianMatrixEigenvalues(
	const marray::Marray<T>& dxx, 
	const marray::Marray<T>& dyy,
	const marray::Marray<T>& dzz,
	const marray::Marray<T>& dxy, 
	const marray::Marray<T>& dxz,
	const marray::Marray<T>& dyz,
	marray::Marray<T> & out
) {
	size_t shapeOut[] = {dxx.shape(0), dyy.shape(1), dzz.shape(2), 3};
	out.resize(marray::SkipInitialization, shapeOut, shapeOut + 4);

	for(size_t z = 0; z < dzz.shape(2); ++z) 
	for(size_t y = 0; y < dyy.shape(1); ++y) 
	for(size_t x = 0; x < dxx.shape(0); ++x) {				
		const T a = dxx  (x, y, z);
		const T b = dxy(x, y, z);
		const T c = dxz(x, y, z);
		const T d = dyy(x, y, z);
		const T e = dyz(x, y, z);
		const T f = dzz(x, y, z);

		// 0 1 2 positions are reserved for 3 real roots. The index for their imaginary part is + 3 (if any).
		vision::computeEigen(a, b, c, d, e, f, out(x, y, z, 0), out(x, y, z, 1), out(x, y, z, 2));
	}
}

} // namespace features

#endif // #ifndef ANDRES_VISION_HESSIAN_MATRIX_HXX
