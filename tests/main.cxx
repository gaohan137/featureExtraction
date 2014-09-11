#include <iostream>
#include <ctime>

#include "marray.hxx"
#include "marray_hdf5.hxx"
#include "meta.hxx"
#include "bilateral-filter.hxx"
#include "gradient-magnitude.hxx"
#include "hessian-matrix.hxx"
#include "structure-tensor.hxx"
#include "partition.hxx"
#include "feature-extraction.hxx"

int main() {

	clock_t t1, t2;
	t1 = clock();

	typedef vision::TypeList<vision::StructureTensorEigenvalues<double>, vision::TypeList<vision::HessianMatrixEigenvalues<double>, vision::TypeList<vision::GradientMagnitude<double>, vision::TypeList<andres::vision::BilateralFilter<double> > > > > typelist;

	vision::FeatureExtraction<double, typelist> extraction;

	extraction.feature<0>().derivativeSigma() = 1;
	extraction.feature<0>().smoothingSigma() = 2;
	extraction.feature<0>().derivativeRadius() = 4;
	extraction.feature<0>().smoothingRadius() = 6;

	extraction.feature<1>().sigma() = 1;
	extraction.feature<1>().radius() = 4;

	extraction.feature<2>().sigma() = 1;
	extraction.feature<2>().radius() = 4;

	//coefficients are fixed for bilateral filter.
	extraction.feature<3>().spatialScale() = 1;
	extraction.feature<3>().intensityScale() = 0.125;
	extraction.feature<3>().radius() = 4;

	std::string dir = "C:/Users/verena/Desktop/data/small-inverted.h5";
	hid_t file = marray::hdf5::openFile(dir);

	vision::Partition<double, typelist> partition(50, dir, "set1", extraction, file);

	marray::hdf5::closeFile(file);

	t2 = clock();
	std::cout<<(double)(t2 - t1) / CLOCKS_PER_SEC<<" seconds"<<std::endl;

	system("pause");

	return 0;
}