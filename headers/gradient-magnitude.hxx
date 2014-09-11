/// Copyright (c) 2012 by Gao Han, Mark Matten, Xin Sun.
/// Copy freely.
#pragma once
#ifndef ANDRES_VISION_GRADIENT_MAGNITUDE_HXX
#define ANDRES_VISION_GRADIENT_MAGNITUDE_HXX

#include <cmath>

#include "marray.hxx"
#include "marray_hdf5.hxx"
#include "convolution.hxx"

namespace vision {

template<class T>
class GradientMagnitude{
public:
	typedef T Value;

	GradientMagnitude();
	GradientMagnitude(const Value, const size_t);

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
GradientMagnitude<T>::GradientMagnitude() 
:	sigma_(static_cast<Value>(1)),
	radius_(4)
{}

template<class T>
inline
GradientMagnitude<T>::GradientMagnitude(
	const Value sigma, 
	const size_t radius
) 
:	sigma_(sigma),
	radius_(radius)
{}

template<class T>
inline typename GradientMagnitude<T>::Value& 
GradientMagnitude<T>::sigma() {
	return sigma_;
}

template<class T>
inline typename GradientMagnitude<T>::Value 
GradientMagnitude<T>::sigma() const {
	return sigma_;
}

template<class T>
inline size_t& 
GradientMagnitude<T>::radius() {
	return radius_;
}

template<class T>
inline size_t 
GradientMagnitude<T>::radius() const {
	return radius_;
}

template<class T>
inline size_t 
GradientMagnitude<T>::margin() const {
	return radius_;
}	

template<class T>
inline size_t 
GradientMagnitude<T>::numberOfValues(
	const size_t dimension
) const {
	return 1;
}

template<class T>
inline void 
GradientMagnitude<T>::compute(
	const marray::View<Value>& in, 
	marray::Marray<Value>& out
) const {
	if (in.dimension() == 2) {
		size_t shape[] = {in.shape(0), in.shape(1)};
		out.resize(marray::SkipInitialization, shape, shape + 2);
		gradientMagnitude2D<T>(sigma_, radius_, in, out);
	}
	else if (in.dimension() == 3) {
		size_t shape[] = {in.shape(0), in.shape(1), in.shape(2)};
		out.resize(marray::SkipInitialization, shape, shape + 3);
		gradientMagnitude3D<T>(sigma_, radius_, in, out);	
	}
	else {
		std::runtime_error("Improper dimension for image.");
	}
}

template<class T>
inline void
gradientMagnitude2D(
	const T sigma,
	const size_t radius,
	const marray::Marray<T>& img,
	marray::Marray<T>& out
) {
	size_t shape[] = {img.shape(0), img.shape(1)};
	out.resize(marray::SkipInitialization, shape, shape + 2);

	marray::Marray<T> dx;
	marray::Marray<T> dy;
	vision::derive(sigma, radius, 0, img, dx);
	vision::derive(sigma, radius, 1, img, dy);

	for (size_t y = 0; y < img.shape(1); ++y)
	for (size_t x = 0; x < img.shape(0); ++x) {
		out(x, y) = sqrt(dx(x, y) * dx(x, y) + dy(x, y) * dy(x, y));
	}
}

template<class T>
inline void
gradientMagnitude3D(
	const T sigma,
	const size_t radius,
	const marray::Marray<T>& img,
	marray::Marray<T>& out
) {
	size_t shape[] = {img.shape(0), img.shape(1), img.shape(2)};
	out.resize(marray::SkipInitialization, shape, shape + 3);

	marray::Marray<T> dx;
	marray::Marray<T> dy;
	marray::Marray<T> dz;
	vision::derive(sigma, radius, 0, img, dx);
	vision::derive(sigma, radius, 1, img, dy);
	vision::derive(sigma, radius, 2, img, dz);

	for (size_t z = 0; z < img.shape(2); ++z)
	for (size_t y = 0; y < img.shape(1); ++y)
	for (size_t x = 0; x < img.shape(0); ++x) {
		out(x, y, z) = sqrt(dx(x, y, z) * dx(x, y, z) + dy(x, y, z) * dy(x, y, z) + dz(x, y, z) * dz(x, y, z));
	}
}

} // namespace features

#endif // #ifndef ANDRES_VISION_GRADIENT_MAGNITUDE_HXX