/// Copyright (c) 2012 by Gao Han, Mark Matten, Xin Sun.
/// Copy freely.
#pragma once
#ifndef ANDRES_VISION_DERIVATIVES_HXX
#define ANDRES_VISION_DERIVATIVES_HXX

#include <cassert>
#include <cmath>

#include "marray.hxx"
#include "marray_hdf5.hxx"
#include "computations.hxx"

namespace vision {

inline double
derivativeOfGaussian(
	const double sigma, 
	const double x
) {
	const double sigma2 = sigma * sigma;
	return -x / (sqrt(2.0 * PI) * sigma2 * sigma) * exp(-0.5 * x * x / sigma2);
}

template<class T>
inline void 
derivativeOfGaussian(
	const double sigma, 
	const size_t radius, 
	marray::Marray<T>& out
) {
	size_t shape[] = {radius * 2 + 1};
	out.resize(marray::SkipInitialization, shape, shape + 1);
	for (size_t i = 0; i <= 2 * radius; ++i) {
		out(i) = static_cast<T>(derivativeOfGaussian(sigma, static_cast<double>(i) - static_cast<double>(radius)));
	}
}

inline double
secondDerivativeOfGaussian(
	const double sigma, 
	const double x
) {
	const double sigma2 = sigma * sigma;
	return 1 / (sqrt(2.0 * PI) * sigma2 * sigma) * ((x * x / sigma2) - 1.0) * exp(-0.5 * x * x / sigma2);
}

template<class T>
inline void 
secondDerivativeOfGaussian(
	const double sigma, 
	const size_t radius, 
	marray::Marray<T>& out
) {
	size_t shape[] = {radius * 2 + 1};
	out.resize(marray::SkipInitialization, shape, shape + 1);
	for (size_t i = 0; i <= 2 * radius; ++i){
		out(i) = static_cast<T>(secondDerivativeOfGaussian(sigma, static_cast<double>(i) - static_cast<double>(radius)));
	}
}

template<class T>
inline void 
derive(
	const T sigma,
	const size_t radius,
	const size_t dimension, // 0 for dx, 1 for dy, 2 for dz.
	const marray::Marray<T>& img,
	marray::Marray<T>& out
) {
	size_t shape[] = {img.shape(0), img.shape(1), 1};
	shape[2] = (img.dimension() == 3 ? img.shape(2) : 1);

	out.resize(marray::SkipInitialization, shape, shape + 3);
	marray::Marray<T> derivativeMask;
	derivativeOfGaussian<T>(sigma, radius, derivativeMask);
	if (dimension == 1) {
		size_t temp1[] = {1, derivativeMask.size(), 1};
		derivativeMask.reshape(temp1, temp1 + 3);
	}
	else if (dimension == 2) {
		size_t temp2[] = {1, 1, derivativeMask.size()};
		derivativeMask.reshape(temp2, temp2 + 3);
	}
	convolve<T>(img, derivativeMask, out);
}

template<class T>
inline void 
secondDerive(
	const T sigma,
	const size_t radius,
	const size_t dimension, // 0 for dx, 1 for dy, 2 for dz.
	const marray::Marray<T>& img,
	marray::Marray<T>& out
) {
	size_t shape[] = {img.shape(0), img.shape(1), 1};
	shape[2] = (img.dimension() == 3 ? img.shape(2) : 1);

	out.resize(marray::SkipInitialization, shape, shape + 3);
	marray::Marray<T> secondDerivativeMask;
	secondDerivativeOfGaussian<T>(sigma, radius, secondDerivativeMask);
	if (dimension == 1) {
		size_t temp1[] = {1, secondDerivativeMask.size(), 1};
		secondDerivativeMask.reshape(temp1, temp1 + 3);
	}
	else if (dimension == 2) {
		size_t temp2[] = {1, 1, secondDerivativeMask.size()};
		secondDerivativeMask.reshape(temp2, temp2 + 3);
	}
	convolve<T>(img, secondDerivativeMask, out);
}

} // namespace vision

#endif // #ifndef ANDRES_VISION_DERIVATIVES_HXX