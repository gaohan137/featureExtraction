/// Copyright (c) 2012 by Gao Han, Mark Matten, Xin Sun.
/// Copy freely.
#pragma once
#ifndef ANDRES_VISION_STRUCTURE_TENSOR_HXX
#define ANDRES_VISION_STRUCTURE_TENSOR_HXX

#include <cmath>

#include "marray.hxx"
#include "marray_hdf5.hxx"
#include "convolution.hxx"
#include "derivatives.hxx"
#include "computations.hxx"

namespace vision {

template<class T>
class StructureTensorEigenvalues{
public:	
	typedef T Value;

	StructureTensorEigenvalues();
	StructureTensorEigenvalues(const Value, const Value, const size_t, const size_t);

	Value& derivativeSigma();
	Value derivativeSigma() const;
	Value& smoothingSigma();
	Value smoothingSigma() const;
	size_t& derivativeRadius();
	size_t derivativeRadius() const;
	size_t& smoothingRadius();
	size_t smoothingRadius() const;
	size_t margin() const;
	size_t numberOfValues(const size_t) const;
	void compute(const marray::View<Value>&, marray::Marray<Value>&) const;

private:
	Value sigma1_;
	Value sigma2_;
	size_t radius1_;
	size_t radius2_;
};

template<class T>
inline
StructureTensorEigenvalues<T>::StructureTensorEigenvalues()
:	sigma1_(static_cast<Value>(1.0)),
	sigma2_(static_cast<Value>(2.0)),
	radius1_(4),
	radius2_(8)
{}

template<class T>
inline
StructureTensorEigenvalues<T>::StructureTensorEigenvalues(
	const Value sigma1, 
	const Value sigma2,
	const size_t radius1, 
	const size_t radius2
)
:	sigma1_(sigma1),
	sigma2_(sigma2),
	radius1_(radius1),
	radius2_(radius2)
{}

template<class T>
inline typename StructureTensorEigenvalues<T>::Value& 
StructureTensorEigenvalues<T>::derivativeSigma() {
	return sigma1_;
}

template<class T>
inline typename StructureTensorEigenvalues<T>::Value 
StructureTensorEigenvalues<T>::derivativeSigma() const {
	return sigma1_;
}

template<class T>
inline typename StructureTensorEigenvalues<T>::Value& 
StructureTensorEigenvalues<T>::smoothingSigma() {
	return sigma2_;
}

template<class T>
inline typename StructureTensorEigenvalues<T>::Value 
StructureTensorEigenvalues<T>::smoothingSigma() const {
	return sigma2_;
}

template<class T>
inline size_t& 
StructureTensorEigenvalues<T>::derivativeRadius() {
	return radius1_;
}

template<class T>
inline size_t 
StructureTensorEigenvalues<T>::derivativeRadius() const {
	return radius1_;
}

template<class T>
inline size_t& 
StructureTensorEigenvalues<T>::smoothingRadius() {
	return radius2_;
}

template<class T>
inline size_t 
StructureTensorEigenvalues<T>::smoothingRadius() const {
	return radius2_;
}

template<class T>
inline size_t 
StructureTensorEigenvalues<T>::margin() const{
	// Changed from 'radius1_ + radius2_' to avoid overlap error
	return 2 * (radius1_ + radius2_) + 1;
}

template<class T>
inline size_t 
StructureTensorEigenvalues<T>::numberOfValues(
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
StructureTensorEigenvalues<T>::compute(
	const marray::View<Value>& in, 
	marray::Marray<Value>& out
) const {
	// create derivative mask
	marray::Marray<T> derivativeMask;
	vision::derivativeOfGaussian<T>(sigma1_, radius1_, derivativeMask);
	// compute derivatives
	marray::Marray<T> dx;
	marray::Marray<T> dy;
	vision::derive<T>(sigma1_, radius1_, 0, in, dx);
	vision::derive<T>(sigma1_, radius1_, 1, in, dy);
	if (in.dimension() == 2) {
		size_t shape[] = {in.shape(0), in.shape(1), 2};
		out.resize(marray::SkipInitialization, shape, shape + 3);
		structureTensorEigenvalues<T>(sigma2_, radius2_, dx, dy, out);
	}
	else if (in.dimension() == 3) {
		size_t shape[] = {in.shape(0), in.shape(1), in.shape(2), 3};
		out.resize(marray::SkipInitialization, shape, shape + 4);
		marray::Marray<T> dz;
		vision::derive<T>(sigma1_, radius1_, 2, in, dz);
		structureTensorEigenvalues<T>(sigma2_, radius2_, dx, dy, dz, out);
	}
	else {
		std::runtime_error("Improper dimension for image.");
	}
}

template<class T>
void 
structureTensorEigenvalues(
	const T sigma,
	const size_t radius,
	const marray::Marray<T>& dx, 
	const marray::Marray<T>& dy,
	marray::Marray<T> & out
) {
	size_t shapeOut[] = {dx.shape(0), dy.shape(1), 2};
	out.resize(marray::SkipInitialization, shapeOut, shapeOut + 3);

	marray::Marray<T> dxSquare = dx * dx;
	marray::Marray<T> dxdy = dx * dy;
	marray::Marray<T> dySquare = dy * dy;
	marray::Marray<T> mask;
	vision::gaussian(sigma, radius, mask);
	vision::convolve(dxSquare, mask);
	vision::convolve(dxdy, mask);
	vision::convolve(dySquare, mask);
	size_t shape[] = {1, mask.size(), 1};
	mask.reshape(shape, shape + 3);
	vision::convolve(dxSquare, mask);
	vision::convolve(dxdy, mask);
	vision::convolve(dySquare, mask);
	for (size_t y = 0; y < dx.shape(1); ++y) 
	for (size_t x = 0; x < dx.shape(0); ++x) {
		const T a = dxSquare(x, y);
		const T b = dxdy(x, y);
		const T c = dxdy(x, y);
		const T d = dySquare(x, y);
		const T B = -(a + d);
		const T C = a * d - b * b;
		vision::roots(1, B, C, out(x, y, 0), out(x, y, 1));
	}
}

template<class T>
void 
structureTensorEigenvalues(
	const T sigma,
	const size_t radius,
	const marray::Marray<T>& dx, 
	const marray::Marray<T>& dy,
	const marray::Marray<T>& dz,
	marray::Marray<T> & out
) {
	size_t shapeOut[] = {dx.shape(0), dy.shape(1), dz.shape(2), 3};
	out.resize(marray::SkipInitialization, shapeOut, shapeOut + 4);

	marray::Marray<T> dxSquare = dx * dx;
	marray::Marray<T> dySquare = dy * dy;
	marray::Marray<T> dzSquare = dz * dz;
	marray::Marray<T> dxdy = dx * dy;
	marray::Marray<T> dxdz = dx * dz;
	marray::Marray<T> dydz = dz * dy;

	marray::Marray<T> mask;
	vision::gaussian(sigma, radius, mask);
	vision::convolve(dxSquare, mask);
	vision::convolve(dySquare, mask);
	vision::convolve(dzSquare, mask);
	vision::convolve(dxdy, mask);
	vision::convolve(dxdz, mask);
	vision::convolve(dydz, mask);
	
	size_t shape[] = {1, mask.size(), 1};
	mask.reshape(shape, shape + 3);
	vision::convolve(dxSquare, mask);
	vision::convolve(dySquare, mask);
	vision::convolve(dzSquare, mask);
	vision::convolve(dxdy, mask);
	vision::convolve(dxdz, mask);
	vision::convolve(dydz, mask);

	size_t newShape[] = {1, 1, mask.size()};
	mask.reshape(newShape, newShape + 3);
	vision::convolve(dxSquare, mask);
	vision::convolve(dySquare, mask);
	vision::convolve(dzSquare, mask);
	vision::convolve(dxdy, mask);
	vision::convolve(dxdz, mask);
	vision::convolve(dydz, mask);

	for(size_t z = 0; z < dx.shape(2); ++z) 
	for(size_t y = 0; y < dx.shape(1); ++y) 
	for(size_t x = 0; x < dx.shape(0); ++x) {				
		const T a = dxSquare(x, y, z);
		const T b = dxdy(x, y, z);
		const T c = dxdz(x, y, z);
		const T d = dySquare(x, y, z);
		const T e = dydz(x, y, z);
		const T f = dzSquare(x, y, z);

		// 0 1 2 positions are reserved for 3 real roots. The index for their imaginary part is + 3 (if any).
		vision::computeEigen(a, b, c, d, e, f, out(x, y, z, 0), out(x, y, z, 1), out(x, y, z, 2));
	}
}

} // namespace features

#endif // #ifndef ANDRES_VISION_STRUCTURE_TENSOR_HXX
