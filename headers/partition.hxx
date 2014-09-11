/// Copyright (c) 2012 by Gao Han, Mark Matten, Xin Sun.
/// Copy freely.
#pragma once
#ifndef ANDRES_VISION_PARTITION_HXX
#define ANDRES_VISION_PARTITION_HXX

#include "marray.hxx"
#include "marray_hdf5.hxx"
#include "feature-extraction.hxx"

namespace vision {

template<class T, class TYPELIST>
class Partition {
public:
	Partition(const size_t, const std::string&, const std::string&, FeatureExtraction<T, TYPELIST>&, const hid_t);

private:
	size_t serialNumber;
	marray::Marray<T> blockContainer;
	size_t originalRow;
	size_t originalColumn;
	size_t originalZ;
	size_t maskSize_;
	size_t blockSize_;
	std::string dir_;
	std::string dataset_;
};

template<class T, class TYPELIST>
Partition<T, TYPELIST>::Partition(
	const size_t blockSize, 
	const std::string& dir,
	const std::string& dataset,
	FeatureExtraction<T, TYPELIST>& extract,
	const hid_t file
) {
	// first time flag
	size_t isFirstTime = 1;
	// load shape
	marray::Vector<size_t> dataShape;
	marray::hdf5::loadShape(file, dataset, dataShape);
	std::string savingDir = dir.substr(0, dir.length()-3) + "_features.h5";
	hid_t fileSaved = marray::hdf5::createFile(savingDir);
	const unsigned char lengthOffset = vision::Length<TYPELIST>::value - 1;
	size_t maskSize = extract.maxMargin<lengthOffset>();
	size_t shape_of_no_use[] = {10, 20};
	marray::Marray<T> temp_marray_for_vector(marray::SkipInitialization, shape_of_no_use, shape_of_no_use + 2);
	std::vector<marray::Marray<T> > marrayVector(lengthOffset + 1, temp_marray_for_vector);

	if (dataShape.size() == 2) {
		for (size_t i = 0; i < dataShape(0); i = i + blockSize - (maskSize - 1)) {
			for (size_t j = 0; j < dataShape(1); j = j + blockSize - (maskSize - 1)) {
				size_t base[] = {i, j};    
				size_t shapeOfBlock[] = {(dataShape(0) - i < blockSize ? dataShape(0) - i : blockSize),
					(dataShape(1) - j < blockSize ? dataShape(1) - j : blockSize)
				};
				marray::hdf5::loadHyperslab(file, dataset, base, base + 2, shapeOfBlock, blockContainer);

				// Apply bilateral filter 4 times
				/*size_t tempFilterShape[] = {blockContainer.shape(0), blockContainer.shape(1)};
				marray::Marray<double> tempFilter(marray::SkipInitialization, tempFilterShape, tempFilterShape + 2);
				marray::Marray<double> tempFilter2(marray::SkipInitialization, tempFilterShape, tempFilterShape + 2);
				marray::Marray<double> tempFilter3(marray::SkipInitialization, tempFilterShape, tempFilterShape + 2);
				marray::Marray<double> tempFilter4(marray::SkipInitialization, tempFilterShape, tempFilterShape + 2);
				extract.feature<3>().compute(blockContainer, tempFilter);
				extract.feature<3>().compute(tempFilter, tempFilter2);
				extract.feature<3>().compute(tempFilter2, tempFilter3);
				extract.feature<3>().compute(tempFilter3, tempFilter4);*/
				
				// Change 'blockContainer' to 'tempFilter4' to apply bilateral filter
				// Extract all features
				extract.extractAll<lengthOffset>(blockContainer, marrayVector);              

				size_t tempBase[] = {(i == 0 ? 0 : (maskSize - 1) - maskSize/2 ), 
					                 (j == 0 ? 0 : (maskSize - 1) - maskSize/2 )};
				size_t tempShape[] = {((i + blockSize) >= dataShape(0) ? (i == 0 ? dataShape(0) - i : dataShape(0) - i - (maskSize -1 - maskSize/2)) : (i == 0 ? blockSize - maskSize / 2 : blockSize - (maskSize - 1))), 
					                  ((j + blockSize) >= dataShape(1) ? (j == 0 ? dataShape(1) - j : dataShape(1) - j - (maskSize -1 - maskSize/2)) : (j == 0 ? blockSize - maskSize / 2 : blockSize - (maskSize - 1)))};     
				size_t vectorLength = 0;
				for (size_t r = 0; r < lengthOffset + 1; ++r) {     
					size_t a1 = 0;
					if (marrayVector[r].dimension() == blockContainer.dimension()) {
						a1 = 1;
					}
					else {
						a1 = marrayVector[r].shape(marrayVector[r].dimension() - 1);
					}
					vectorLength = vectorLength + a1;
				}
				// create file for each feature value
				size_t size[] = {dataShape(0), dataShape(1)};
				marray::Marray<double> empty(size, size + 2, 0);
				if (isFirstTime == 1) {
					long double pre_setNumber = 1;
					for (size_t o = 0; o < vectorLength; ++o) {
						marray::hdf5::save(fileSaved, std::to_string(pre_setNumber), empty);
						++pre_setNumber;
					}
					pre_setNumber = 1;
					isFirstTime = 0;
				}
				std::vector<marray::View<T> > viewVector(vectorLength);     
				std::vector<marray::Marray<T> > marrayFinalVector(vectorLength);
				int temp_index = 0;
				for (size_t d1 = 0; d1 < lengthOffset + 1; ++d1)
					for (size_t d2 = 0; d2 < (marrayVector[d1].dimension() == blockContainer.dimension() ? 1 : marrayVector[d1].shape(marrayVector[d1].dimension() - 1)); ++d2) {

						if (marrayVector[d1].dimension() != blockContainer.dimension()) {
							viewVector[temp_index] =  marrayVector[d1].boundView(marrayVector[d1].dimension() - 1, d2);
						}
						else if (marrayVector[d1].dimension() == blockContainer.dimension()) {
							marray::View<T> tempV(marrayVector[d1]);
							viewVector[temp_index] = tempV;
						}
						viewVector[temp_index].view(tempBase, tempShape, viewVector[temp_index]);              
						marray::Marray<T> tempM(viewVector[temp_index]);
						marrayFinalVector[temp_index] = tempM;
						++temp_index;
					}

					size_t tempXY[] = {(i == 0 ? i : i + maskSize - 1 - maskSize/2 ), (j == 0 ? j : j + maskSize - 1 - maskSize/2) };     

					long double setNumber = 1;
					for (size_t v = 0; v < vectorLength; ++v) {    
						marray::hdf5::saveHyperslab(fileSaved, std::to_string(setNumber), tempXY, tempXY + 2, tempShape, marrayFinalVector[v]);
						++setNumber;
					}
					if (dataShape(1) - j <= blockSize) {
						break;
					}
			}
			if (dataShape(0) - i <= blockSize) {
				break;
			}
		}
	}
	else if (dataShape.size() == 3) {
		for (size_t i = 0; i < dataShape(0); i = i + blockSize - (maskSize - 1)) {
			for (size_t j = 0; j < dataShape(1); j = j + blockSize - (maskSize - 1)) {
				for (size_t k = 0; k < dataShape(2); k = k + blockSize - (maskSize - 1)) {
					size_t base[] = {i, j, k};    
					size_t shapeOfBlock[] = {(dataShape(0) - i < blockSize ? dataShape(0) - i : blockSize),
						(dataShape(1) - j < blockSize ? dataShape(1) - j : blockSize),
						(dataShape(2) - k < blockSize ? dataShape(2) - k : blockSize)
					};
					marray::hdf5::loadHyperslab(file, dataset, base, base + 3, shapeOfBlock, blockContainer);
					

					// Apply bilateral filter 4 times
					/*size_t tempFilterShape[] = {blockContainer.shape(0), blockContainer.shape(1), blockContainer.shape(2)};
					marray::Marray<double> tempFilter(marray::SkipInitialization, tempFilterShape, tempFilterShape + 3);
					marray::Marray<double> tempFilter2(marray::SkipInitialization, tempFilterShape, tempFilterShape + 3);
					marray::Marray<double> tempFilter3(marray::SkipInitialization, tempFilterShape, tempFilterShape + 3);
					marray::Marray<double> tempFilter4(marray::SkipInitialization, tempFilterShape, tempFilterShape + 3);
					extract.feature<3>().compute(blockContainer, tempFilter);
					extract.feature<3>().compute(tempFilter, tempFilter2);
					extract.feature<3>().compute(tempFilter2, tempFilter3);
					extract.feature<3>().compute(tempFilter3, tempFilter4);*/
				
					// Change 'blockContainer' to 'tempFilter4' to apply bilateral filter
					// Extract all features 
					extract.extractAll<lengthOffset>(blockContainer, marrayVector);
					// Below is the base and shape relative to the Marray<T>.
					size_t tempBase[] = {(i == 0 ? 0 : (maskSize - 1) - maskSize/2), 
						                 (j == 0 ? 0 : (maskSize - 1) - maskSize/2), 
						                 (k == 0 ? 0 : (maskSize - 1) - maskSize/2)};
					size_t tempShape[] = {((i + blockSize) >= dataShape(0) ? (i == 0 ? dataShape(0) - i : dataShape(0) - i - (maskSize -1 - maskSize/2)) : (i == 0 ? blockSize - maskSize/2 : blockSize - (maskSize - 1))),
						                  ((j + blockSize) >= dataShape(1) ? (j == 0 ? dataShape(1) - j : dataShape(1) - j - (maskSize -1 - maskSize/2)) : (j == 0 ? blockSize - maskSize/2 : blockSize - (maskSize - 1))),
						                  ((k + blockSize) >= dataShape(2) ? (k == 0 ? dataShape(2) - k : dataShape(2) - k - (maskSize -1 - maskSize/2)) : (k == 0 ? blockSize - maskSize/2 : blockSize - (maskSize - 1)))};
					size_t vectorLength = 0;
					for (size_t r = 0; r < lengthOffset + 1; ++r) {     
						size_t a1 = 0;
						if (marrayVector[r].dimension() == blockContainer.dimension()) {
							a1 = 1;
						}
						else {
							a1 = marrayVector[r].shape(marrayVector[r].dimension() - 1);
						}
						vectorLength = vectorLength + a1;
					}
					// create file for each feature value
					size_t size[] = {dataShape(0), dataShape(1), dataShape(2)};
					marray::Marray<double> empty(size, size + 3, 0);
					if (isFirstTime == 1) {
						long double pre_setNumber = 1;
						for (size_t o = 0; o < vectorLength; ++o) {
							marray::hdf5::save(fileSaved, std::to_string(pre_setNumber), empty);
							++pre_setNumber;
						}
						// reset isFirstTime flag to 0
						isFirstTime = 0;
					}
					// Start splitting multi-Marray<T> into Marray<T>s that have the same size of original block.
					std::vector<marray::View<T> > viewVector(vectorLength);
					std::vector<marray::Marray<T> > marrayFinalVector(vectorLength);    
					int temp_index = 0;
					for (size_t d1 = 0; d1 < lengthOffset + 1; ++d1)
						for (size_t d2 = 0; d2 < (marrayVector[d1].dimension() == blockContainer.dimension() ? 1 : marrayVector[d1].shape(marrayVector[d1].dimension() - 1)); ++d2) {

							if (marrayVector[d1].dimension() != blockContainer.dimension()) {
								viewVector[temp_index] =  marrayVector[d1].boundView(marrayVector[d1].dimension() - 1, d2);
							}
							else if (marrayVector[d1].dimension() == blockContainer.dimension()) {
								marray::View<T> tempV(marrayVector[d1]);
								viewVector[temp_index] = tempV;
							}
							viewVector[temp_index].view(tempBase, tempShape, viewVector[temp_index]);              
							marray::Marray<T> tempM(viewVector[temp_index]);
							marrayFinalVector[temp_index] = tempM;
							++temp_index;
						}     
						// adjusting coordinates.
						size_t tempXY[] = {(i == 0 ? i : i + maskSize - 1 - maskSize/2 ), 
							(j == 0 ? j : j + maskSize - 1 - maskSize/2 ), 
							(k == 0 ? k : k + maskSize - 1 - maskSize/2 ) };     

						long double setNumber = 1;
						for (size_t v = 0; v < vectorLength; ++v) {    
							marray::hdf5::saveHyperslab(fileSaved, std::to_string(setNumber), tempXY, tempXY + 3, tempShape, marrayFinalVector[v]);
							++setNumber;
						}
						if (dataShape(2) - k <= blockSize) {
							break;
						}
				}
				if (dataShape(1) - j <= blockSize) {
					break;
				}
			}
			if (dataShape(0) - i <= blockSize) {
				break;
			}
		}
	}
	else {
		throw std::runtime_error("Improper dimension.");
	}	
}

} // namespace partition

#endif // #ifndef ANDRES_VISION_PARTITION_HXX
