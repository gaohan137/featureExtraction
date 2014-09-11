/// Copyright (c) 2012 by Bjoern Andres, Gao Han, Mark Matten, Xin Sun.
/// Copy freely.
#pragma once
#ifndef FEATURES_META_HXX
#define FEATURES_META_HXX

namespace vision {

struct TypeListEnd {
};

template<class HEAD, class TAIL = TypeListEnd>
struct TypeList {
    typedef HEAD Head;
    typedef TAIL Tail;
};

template<class TYPELIST, unsigned char N, unsigned char K = 0>
struct SubTypeList {
	typedef typename SubTypeList<typename TYPELIST::Tail, N, K + 1>::TypeListType TypeListType;
	typedef typename TypeListType::Head ValueType;
};

template<class TYPELIST, unsigned char N>
struct SubTypeList<TYPELIST, N, N> {
	typedef TYPELIST TypeListType;
	typedef typename TypeListType::Head ValueType;
};

template<class TYPELIST>
struct Field 
: public Field<typename TYPELIST::Tail> {
    typedef TYPELIST TypeListType;
    typename TypeListType::Head item_;
};

template<>
struct Field<TypeListEnd> {
};

// Find the length of a TypeList
template<class TYPELIST> 
struct Length;

template<> 
struct Length<TypeListEnd> {
	static const unsigned char value = 0;
};

template<class H, class T> 
struct Length<TypeList<H, T> > {
	static const unsigned char value = 1 + Length<T>::value;
};

// Add a type to the beginning of a TypeList
template<class NEWTYPE, class OLDLIST>
struct TypeListPushFront {
	typedef TypeList<NEWTYPE, OLDLIST> newType;
};

// Add a type in the middle of a TypeList
template<class TYPELIST, class NEWTYPE, unsigned char N, unsigned char K = 1>
struct AddMiddle;

template<class H, class T, class NEWTYPE, unsigned char N>
struct AddMiddle<TypeList<H, T>, NEWTYPE, N, N> {
	typedef TypeList<H, TypeList<NEWTYPE, T> >newList;
};

template<class H, class T, class NEWTYPE, unsigned char N, unsigned char K>
struct AddMiddle<TypeList<H, T>, NEWTYPE, N, K> {
	typedef TypeList<H, typename AddMiddle<T, NEWTYPE, N, K + 1>::newList> newList;
};

// Add a TypeList to the end of another TypeList
template<class OLDLIST, class ADDEDLIST> 
struct AddEnd {
};

template<class ADDEDLIST>
struct AddEnd<TypeListEnd, ADDEDLIST> {
	typedef ADDEDLIST newList;
};

template<class H, class T, class ADDEDLIST> 
struct AddEnd<TypeList<H,T>, ADDEDLIST> {
	typedef TypeList<H, typename AddEnd<T, ADDEDLIST>::newList> newList;
};

// Count the number of times a certain type appears in a TypeList
template<class TYPELIST, class TYPE>
struct NumOfType {
};

template<class T, class TYPE>
struct NumOfType<TypeList<TYPE, T>, TYPE> {
	enum {counter = 1 + NumOfType<T, TYPE>::counter };
};

template<class TYPE>
struct NumOfType<TypeList<TYPE,TypeListEnd>,TYPE> {
	enum { counter = 1 };
};

template<class H, class TYPE>
struct NumOfType<TypeList<H, TypeListEnd>, TYPE> {
	enum { counter = 0 };
};

template<class H,class T, class TYPE>
struct NumOfType<TypeList<H,T>, TYPE> {
	enum {counter = 0 + NumOfType<T, TYPE>::counter };
};

// Find the index of the first time a certain type appears in the TypeList
template<class TYPELIST, class typeSearch> struct SearchList;

template<class typeSearch>
struct SearchList<TypeListEnd, typeSearch> {
	static const unsigned char value = 0;
}; 

template<class typeSearch, class TAIL>
struct SearchList<TypeList<typeSearch, TAIL>, typeSearch> {
	static const unsigned char value = 0;
};

template<class HEAD, class TAIL, class typeSearch>
struct SearchList<TypeList<HEAD, TAIL>, typeSearch> {
	static const unsigned char value = 1 + SearchList<TAIL, typeSearch>::value;
};

// Copy values from old field to new field
template<class NEWFIELD, class OLDFIELD>
struct CopyData {
};

template<class H1, class T1, class H2, class T2>
struct CopyData<Field<TypeList<H1, T1> >, Field<TypeList<H2, T2> > > {
	void copy(Field<TypeList<H1, T1> > & newF, Field<TypeList<H2, T2> > & oldF) {
		newF.item_ = oldF.item_;
		Field<T1> & ref1 = (Field<T1>&)newF;
		Field<T2> & ref2 = (Field<T2>&)oldF;
		CopyData<Field<T1>, Field<T2> > tempCopy; 
		tempCopy.copy(ref1, ref2);  
	}
};

template<class H, class T>
struct CopyData<Field<TypeList<H, T> >, Field<TypeListEnd> > {
	void copy(Field<TypeList<H, T> > & newF, Field<TypeListEnd> & oldF) {
		// do nothing
	}
};

// Access the Field struct
template<class FIELD, unsigned char N>
inline typename SubTypeList<typename FIELD::TypeListType, N>::ValueType&
accessField(FIELD& field) {
	typedef typename SubTypeList<typename FIELD::TypeListType, N>::TypeListType TypeListType;
    return static_cast<Field<TypeListType>*>(&field)->item_;
}

// Return reference of a Field which contains the new type passed in by the user
template<class NEWTYPE, class OLDFIELD>
Field<typename AddEnd<typename OLDFIELD::TypeListType, TypeList<NEWTYPE> >::newList>&
autoAdd(OLDFIELD& oldfield) {
	typedef typename AddEnd<typename OLDFIELD::TypeListType,TypeList<NEWTYPE> >::newList newTypeList;
	typedef Field<newTypeList> newField;
	newField newfield;
	CopyData<newField, OLDFIELD> tempCopy;
	tempCopy.copy(newfield, oldfield);
	return newfield;
}

} // namespace vision

#endif // #ifndef FEATURES_META_HXX
