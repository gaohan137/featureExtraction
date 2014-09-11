/// Copyright (c) 2012 by Gao Han, Mark Matten, Xin Sun.
/// Copy freely.
#pragma once
#ifndef ANDRES_VISION_FEATURE_EXTRACTION_HXX
#define ANDRES_VISION_FEATURE_EXTRACTION_HXX

#include "meta.hxx"

namespace vision {

template<class T, class TYPELIST>
class FeatureExtraction {
public:
	typedef TYPELIST TypeList;
   	typedef typename vision::Field<TypeList> FeatureField;

	FeatureExtraction(){}
	
	template<unsigned char N>
	typename vision::SubTypeList<TypeList, N>::ValueType& feature() {
		return vision::accessField<FeatureField, N>(field_);
	}

	template<unsigned char N>
	inline void extractAll(const marray::View<T>& in, std::vector<marray::Marray<T> >& out) {
		extractAll<N - 1>(in, out);
		vision::accessField<FeatureField, N>(field_).compute(in, out[N]);
	}

	template<>
	inline void extractAll<0>(const marray::View<T>& in, std::vector<marray::Marray<T> >& out) {	
        vision::accessField<FeatureField, 0>(field_).compute(in, out[0]);
	}

	template<unsigned char N>
	inline size_t maxMargin(const size_t margin = 0) {
		const size_t temp = vision::accessField<FeatureField, N>(field_).margin() > margin ? vision::accessField<FeatureField, N>(field_).margin() : margin;
		return maxMargin<N - 1>(temp);
	}

	template<>
	inline size_t maxMargin<0>(const size_t margin) {
		return vision::accessField<FeatureField, 0>(field_).margin() > margin ? vision::accessField<FeatureField, 0>(field_).margin() : margin;
	}

private:
	FeatureField field_;
};

} // namespace vision

#endif // #ifndef ANDRES_VISION_FEATURE_EXTRACTION_HXX