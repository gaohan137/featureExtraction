/// Copyright (c) 2012 by Bjoern Andres, Gao Han, Mark Matten, Xin Sun.
/// Copy freely.
#pragma once
#ifndef ANDRES_VISION_BILATERAL_HXX
#define ANDRES_VISION_BILATERAL_HXX

#include <stdlib.h>
#include <math.h>
# if defined(__APPLE__)
# else
#     include <malloc.h>
# endif

#define mymax(a,b) (((a) > (b)) ? (a) : (b))
#define mymin(a,b) (((a) < (b)) ? (a) : (b))

namespace andres {
namespace vision {

template<class T>
class BilateralFilter {
public:
	typedef T Value;

	BilateralFilter();
	BilateralFilter(const Value, const Value, const size_t);

	Value& spatialScale();
	Value spatialScale() const;
	Value& intensityScale();
	Value intensityScale() const;
	size_t& radius();
	size_t radius() const;
	size_t margin() const;
	size_t numberOfValues(const size_t) const;
	void compute(const marray::View<Value>&, marray::Marray<Value>&) const;

private:
	Value spatialScale_;
	Value intensityScale_;
	size_t radius_;
};

template<class T>
inline
BilateralFilter<T>::BilateralFilter() 
:	spatialScale_(static_cast<Value>(1)),
	intensityScale_(static_cast<Value>(0.5)),
	radius_(4)
{}

template<class T>
inline
BilateralFilter<T>::BilateralFilter(
	const Value spatialScale, 
	const Value intensityScale,
	const size_t radius
) 
:	spatialScale_(spatialScale),
	intensityScale_(intensityScale),
	radius_(radius)
{}

template<class T>
inline typename BilateralFilter<T>::Value& 
BilateralFilter<T>::spatialScale() {
	return spatialScale_;
}

template<class T>
inline typename BilateralFilter<T>::Value
BilateralFilter<T>::spatialScale() const {
	return spatialScale_;
}

template<class T>
inline typename BilateralFilter<T>::Value& 
BilateralFilter<T>::intensityScale() {
	return intensityScale_;
}

template<class T>
inline typename BilateralFilter<T>::Value
BilateralFilter<T>::intensityScale() const {
	return intensityScale_;
}

template<class T>
inline size_t& 
BilateralFilter<T>::radius() {
	return radius_;
}

template<class T>
inline size_t 
BilateralFilter<T>::radius() const {
	return radius_;
}

template<class T>
inline size_t 
BilateralFilter<T>::margin() const {
	return radius_;
}

template<class T>
inline size_t 
BilateralFilter<T>::numberOfValues(
	const size_t dimension
) const {
	return 1;
}

template<class T>
inline void 
BilateralFilter<T>::compute(
	const marray::View<Value>& in, 
	marray::Marray<Value>& out
) const {
	if (in.dimension() == 2) {
		size_t shape[] = {in.shape(0), in.shape(1)};
		out.resize(marray::SkipInitialization, shape, shape + 2);
		bilateral2<T>(&in(0), in.shape(0), in.shape(1), spatialScale_, intensityScale_, radius_, &out(0));
	}
	else if (in.dimension() == 3) {
		size_t shape[] = {in.shape(0), in.shape(1), in.shape(2)};
		out.resize(marray::SkipInitialization, shape, shape + 3);
		bilateral3<T>(&in(0), in.shape(0), in.shape(1), in.shape(2), spatialScale_, intensityScale_, radius_, &out(0));
	}
	else {
		std::runtime_error("Improper dimension for image.");
	}
}

// C-like implementation of a 2-dimensional bilateral filter
//
// pre-conditions:
// - enough memory has been allocated for out, e.g. using
//   malloc(nx*ny*nz*sizeof(VoxelType))
// - out has been filled with zeros
//
template<class VoxelType>
void 
bilateral2(
    const VoxelType* in,
    const ptrdiff_t nx,
    const ptrdiff_t ny,
    const VoxelType spatialScale,
    const VoxelType intensityScale,
    const ptrdiff_t radius,
    VoxelType* out
) {
    const ptrdiff_t size  = 2 * radius + 1;
    const ptrdiff_t size2 = size * size;
    const size_t    M     = size2 * sizeof(VoxelType);
    const ptrdiff_t nxy   = nx * ny;
    const VoxelType rho   = intensityScale * intensityScale;
    const VoxelType p     = -2 * spatialScale * spatialScale;

    VoxelType *gaussFilter = (VoxelType*) malloc(M);
    VoxelType *bilateralFilter = (VoxelType*) malloc(M);

    // gauss filter pre-computation
    for(ptrdiff_t x=-radius; x<=radius; ++x)
    for(ptrdiff_t y=-radius; y<=radius; ++y) {
        gaussFilter[(x+radius) + size*(y+radius)]
            = exp( (x*x + y*y)/p ) ;
    }

    // diffusion (push)
    for(ptrdiff_t x0=0; x0<nx; ++x0)
    for(ptrdiff_t y0=0; y0<ny; ++y0) {
        ptrdiff_t j0 = x0 + nx*y0 ;
        VoxelType sum = 0;
        // ***
        // compute filter mask
        for(ptrdiff_t x=mymax(x0-radius, 0); x<=mymin(x0+radius, nx-1); ++x)
        for(ptrdiff_t y=mymax(y0-radius, 0); y<=mymin(y0+radius, ny-1); ++y) {
            ptrdiff_t j = x + nx*y;
            ptrdiff_t xr = x - x0 + radius;
            ptrdiff_t yr = y - y0 + radius;
            ptrdiff_t q = xr + size*yr;
            VoxelType val = gaussFilter[q] * ( 1.0f / (1.0f + (in[j]-in[j0])*(in[j]-in[j0])/rho) );

            bilateralFilter[q] = val;
            sum += val;
        }
        // normalize filter mask
        for(ptrdiff_t x=mymax(x0-radius, 0); x<=mymin(x0+radius, nx-1); ++x)
        for(ptrdiff_t y=mymax(y0-radius, 0); y<=mymin(y0+radius, ny-1); ++y) {
            ptrdiff_t xr = x - x0 + radius;
            ptrdiff_t yr = y - y0 + radius;
            ptrdiff_t q = xr + size*yr;
            bilateralFilter[q] /= sum;
        }

        // ***
        // diffuse
		out[j0] = 0;
        for(ptrdiff_t x=mymax(x0-radius, 0); x<=mymin(x0+radius, nx-1); ++x)
        for(ptrdiff_t y=mymax(y0-radius, 0); y<=mymin(y0+radius, ny-1); ++y) {
            ptrdiff_t j = x + nx*y;
            ptrdiff_t xr = x - x0 + radius;
            ptrdiff_t yr = y - y0 + radius;
            ptrdiff_t q = xr + size*yr;
            out[j0] += (in[j] * bilateralFilter[q]);
        }
    }

    // clean-up
    free(gaussFilter);
    free(bilateralFilter);
}

// C-like implementation of a 3-dimensional bilateral filter
//
// pre-conditions:
// - enough memory has been allocated for out, e.g. using
//   malloc(nx*ny*nz*sizeof(VoxelType))
// - out has been filled with zeros
//
template<class VoxelType>
void 
bilateral3(
    const VoxelType* in,
    const ptrdiff_t nx,
    const ptrdiff_t ny,
    const ptrdiff_t nz,
    const VoxelType spatialScale,
    const VoxelType intensityScale,
    const ptrdiff_t radius,
    VoxelType* out
) {
    const ptrdiff_t size  = 2 * radius + 1;
    const ptrdiff_t size2 = size * size;
    const ptrdiff_t size3 = size2 * size;
    const size_t    M     = size3 * sizeof(VoxelType);
    const ptrdiff_t nxy   = nx * ny;
    const VoxelType rho   = intensityScale * intensityScale;
    const VoxelType p     = -2 * spatialScale * spatialScale;

    VoxelType *gaussFilter = (VoxelType*) malloc(M);
    VoxelType *bilateralFilter = (VoxelType*) malloc(M);

    // gauss filter pre-computation
    for(ptrdiff_t z=-radius; z<=radius; ++z)
    for(ptrdiff_t y=-radius; y<=radius; ++y)
    for(ptrdiff_t x=-radius; x<=radius; ++x) {
        gaussFilter[(x+radius) + size*(y+radius) + size2*(z+radius)]
            = exp( (x*x + y*y + z*z)/p ) ;
    }

    // diffusion (push)
    for(ptrdiff_t z0=0; z0<nz; ++z0)
    for(ptrdiff_t y0=0; y0<ny; ++y0)
    for(ptrdiff_t x0=0; x0<nx; ++x0) {
        ptrdiff_t j0 = x0 + nx*y0 + nxy*z0;
        VoxelType sum = 0;
        // ***
        // compute filter mask
        for(ptrdiff_t z=mymax(z0-radius, 0); z<=mymin(z0+radius, nz-1); ++z)
        for(ptrdiff_t y=mymax(y0-radius, 0); y<=mymin(y0+radius, ny-1); ++y)
        for(ptrdiff_t x=mymax(x0-radius, 0); x<=mymin(x0+radius, nx-1); ++x) {
            ptrdiff_t j = x + nx*y + nxy*z;
            ptrdiff_t xr = x - x0 + radius;
            ptrdiff_t yr = y - y0 + radius;
            ptrdiff_t zr = z - z0 + radius;
            ptrdiff_t q = xr + size*yr + size2*zr;
            VoxelType val = gaussFilter[q] * ( 1.0f / (1.0f + (in[j]-in[j0])*(in[j]-in[j0])/rho) );

            bilateralFilter[q] = val;
            sum += val;
        }
        // normalize filter mask
        for(ptrdiff_t z=mymax(z0-radius, 0); z<=mymin(z0+radius, nz-1); ++z)
        for(ptrdiff_t y=mymax(y0-radius, 0); y<=mymin(y0+radius, ny-1); ++y)
        for(ptrdiff_t x=mymax(x0-radius, 0); x<=mymin(x0+radius, nx-1); ++x) {
            ptrdiff_t xr = x - x0 + radius;
            ptrdiff_t yr = y - y0 + radius;
            ptrdiff_t zr = z - z0 + radius;
            ptrdiff_t q = xr + size*yr + size2*zr;
            bilateralFilter[q] /= sum;
        }

        // ***
        // diffuse
		out[j0] = 0;
        for(ptrdiff_t z=mymax(z0-radius, 0); z<=mymin(z0+radius, nz-1); ++z)
        for(ptrdiff_t y=mymax(y0-radius, 0); y<=mymin(y0+radius, ny-1); ++y)
        for(ptrdiff_t x=mymax(x0-radius, 0); x<=mymin(x0+radius, nx-1); ++x) {
            ptrdiff_t j = x + nx*y + nxy*z;

            ptrdiff_t xr = x - x0 + radius;
            ptrdiff_t yr = y - y0 + radius;
            ptrdiff_t zr = z - z0 + radius;
            ptrdiff_t q = xr + size*yr + size2*zr;

            out[j0] += (in[j] * bilateralFilter[q]);
        }
    }

    // clean-up
    free(gaussFilter);
    free(bilateralFilter);
}

} // namespace vision
} // namespace andres

#endif // #ifndef ANDRES_VISION_BILATERAL_HXX
