/// Copyright (c) 2012 by Gao Han, Mark Matten, Xin Sun.
/// Copy freely.
#pragma once
#ifndef ANDRES_VISION_CONVOLUTION_HXX
#define ANDRES_VISION_CONVOLUTION_HXX

#include <cassert>
#include <cmath>

#include "marray.hxx"
#include "marray_hdf5.hxx"

namespace vision {

template<class T>
inline void 
normalize(
	const marray::View<T>& in, 
	marray::View<T>& out
) {
	assert(in.size() == out.size());
	T sum = 0;
	for(size_t x = 0; x < in.size(); ++x) {
		sum += in(x);
	}
	for(size_t x = 0; x < in.size(); ++x) {
		out(x) = in(x) / sum;
	}
}

template<class T>
inline void 
normalize(
	marray::View<T>& in
) {
	T sum = 0;
	for(size_t x = 0; x < in.size(); ++x) {
		sum += in(x);
	}
	for(size_t x = 0; x < in.size(); ++x) {
		in(x) /= sum;
	}
}

template<class T>
inline void 
convolve(
	const marray::Marray<T>& img, 
	const marray::Marray<T>& mask, 
	marray::Marray<T>& out
) {
	if (img.overlaps(out)) {
		throw std::runtime_error("Input and output overlap."); 
	}
	if (img.dimension() > 3) {
		throw std::runtime_error("This function is not implemented for dimension > 3."); 
	}
	marray::View<T> view(img);
	if (view.dimension() == 2){
		size_t shape[] = {img.shape(0), img.shape(1), 1};
		view.reshape(shape, shape + 3);
	}
	else if (view.dimension() == 1) {
		size_t shape[] = {img.shape(0), 1, 1};
		view.reshape(shape, shape + 3);
	}
	out.resize(marray::SkipInitialization, view.shapeBegin(), view.shapeEnd()); 

	marray::View<T> maskView(mask);
	if (maskView.dimension() == 2){
		size_t shape[] = {mask.shape(0), mask.shape(1), 1};
		maskView.reshape(shape, shape + 3);
	}
	else if (maskView.dimension() == 1) {
		size_t shape[] = {mask.shape(0), 1, 1};
		maskView.reshape(shape, shape + 3);
	}

	for(size_t z = 0; z < view.shape(2); ++z)
    for(size_t y = 0; y < view.shape(1); ++y)
    for(size_t x = 0; x < view.shape(0); ++x) {
		out(x, y, z) = 0;
        for(size_t c = 0; c < maskView.shape(2); ++c)
        for(size_t b = 0; b < maskView.shape(1); ++b)
        for(size_t a = 0; a < maskView.shape(0); ++a) {           
            size_t row;
            size_t col;
            size_t zaxis;
            size_t offset1 = (maskView.shape(0) - (maskView.shape(0) - 1) % 2) / 2;  
            size_t offset2 = (maskView.shape(1) - (maskView.shape(1) - 1) % 2) / 2;  
            size_t offset3 = (maskView.shape(2) - (maskView.shape(2) - 1) % 2) / 2;  
            if (x + a < offset1) {
                row = offset1 - x - a;
            }
            else if (x + a - offset1 > view.shape(0) - 1) {
                row = 2 * (view.shape(0) - 1) - (x + a -offset1);
            }
            else {
                row = x + a - offset1;
            }

            if (y + b < offset2) {
                col = offset2 - y - b;
            }
            else if (y + b - offset2 > view.shape(1) - 1) {
                col = 2 * (view.shape(1) - 1) - (y + b - offset2);
            }
            else {
                col = y + b - offset2;
            }

            if (z + c < offset3) {
                zaxis = offset3 - z - c;
            }
            else if (z + c - offset3 > view.shape(2) -1) {
                zaxis = 2 * (view.shape(2) - 1) - (z + c - offset3);
            }
            else {
                zaxis = z + c - offset3;
            }
			out(x, y, z) += maskView(a, b, c) * view(row, col, zaxis);
		}
	}

	if(img.dimension() != 3) {
		out.reshape(img.shapeBegin(), img.shapeEnd());
	}
}

template<class T>
inline void 
convolve( 
	marray::Marray<T>& img, 
	const marray::Marray<T>& mask 
) {
	marray::Marray<T> tmp; 
	convolve(img, mask, tmp);
	img = tmp;
}

} // namespace vision

#endif // #ifndef ANDRES_VISION_CONVOLUTION_HXX