/// Copyright (c) 2012 by Gao Han, Mark Matten, Xin Sun.
/// Copy freely.
#pragma once
#ifndef ANDRES_VISION_COMPUTATIONS_HXX
#define ANDRES_VISION_COMPUTATIONS_HXX

#include "marray.hxx"
#include "marray_hdf5.hxx"

namespace vision {

const double PI = 3.141592653589793;

inline double 
gaussian(
	const double sigma, 
	const double x
) {
	const double sigma2 = sigma * sigma;
	return 1 / (sqrt(2.0 * PI) * sigma) * exp(-0.5 * x * x / sigma2);
}

template<class T>
inline void 
gaussian(
	const double sigma, 
	const size_t radius, 
	marray::Marray<T>& out
) {
	size_t shape[] = {radius * 2 + 1};
	out.resize(marray::SkipInitialization, shape, shape + 1);
	for (size_t i = 0; i <= 2 * radius; ++i) {
		out(i) = static_cast<T>(gaussian(sigma, static_cast<double>(i) - static_cast<double>(radius)));
	}
}

// solves ax^2 + bx + c = 0.
inline int 
roots(
	const double a, const double b, const double c, 
	double& real1, double& real2
) {  
	if (a == 0) {
		throw std::runtime_error("It's a linear function");
	}
	real1 = 0.5 * ((-b) + sqrt(b * b - 4 * c));
	real2 = 0.5 * ((-b) - sqrt(b * b - 4 * c));
	return 0;
}

// solves ax^3 + bx^2 + cx + d = 0.
inline int 
roots(
	const double a, const double b, const double c, const double d, 
	double& real1, double& real2, double& real3
) {  
	double imag1, imag2, imag3;
	if (a == 0) {
		if(b == 0){
			throw std::runtime_error("It's a linear function.");
		}
	}
	const double p = (3.0 * a * c - b * b) / (3 * a * a); 
	const double q = (2.0 * pow(b, 3.0) - 9 * a * b * c + 27.0 * a * a * d) / (27.0 * pow(a, 3.0)); 
	const double r = b / (3.0 * a); 
	const double h = pow(q / 2.0, 2.0) + pow(p / 3.0, 3.0); 
	const double g = sqrt(h); 
	if (h >= 0) { 
		double u, v;
		if (-q / 2.0 + g < 0) { 
			u = -pow(fabs(-q / 2.0 + g), 1.0 / 3.0); 
		}
		else {
			u = pow((-q / 2.0 + g), 1.0 / 3.0); 
		}
		if (-q / 2.0 - g < 0) {
			v = -pow(fabs(-q / 2.0 - g), 1.0 / 3.0); 
		}
		else {
			v = -pow((-q / 2.0 - g), 1.0 / 3.0);
		}
		if (h == 0) { 
			real1 = u + v - r;         
			imag1 = 0; 
			real2 = -(u + v) / 2 - r;     
			imag2 = 0; 
			real3 = -(u + v) / 2 - r;     
			imag3 = 0; 
		} 
		else { 
			real1 = u + v - r;     
			imag1 = 0; 
			real2 = -(u + v) / 2;   
			imag2 = sqrt(3.0) * (u - v) / 2; 
			real3 = -(u + v) / 2;   
			imag3 = -sqrt(3.0) * (u - v) / 2; 
		} 
	} 
	else {    
		const double fai = acos((-q / 2) / (sqrt(pow(fabs(p), 3) / 27))); 
		real1 = 2 * sqrt(fabs(p) / 3.0) * cos(fai / 3.0) - r; 
		real2 = -2 * sqrt(fabs(p) / 3.0) * cos((fai + PI) / 3.0) - r; 
		real3 = -2 * sqrt(fabs(p) / 3.0) * cos((fai - PI) / 3.0) - r; 
		imag1 = 0;   
		imag2 = 0;   
		imag3 = 0; 
	} 
	return 0;
} 


// Format for input: {a, b, c
//					 b, d, e
//					 c, e, f}
inline int 
computeEigen(
	const double a, const double b, const double c, const double d, const double e, const double f,
	double& real1, double& real2, double& real3
) {
		const double A = -1;
		const double B = a + d + f;
		const double C = b * b + c * c + e * e - a * d - a * f - d * f;
		const double D = a * d * f + 2 * b * c * e - a * e * e - b * b * f - c * c * d;
		return roots(A, B, C, D, real1, real2, real3);
}

} // namespace vision

#endif // #ifndef ANDRES_VISION_COMPUTATIONS_HXX
