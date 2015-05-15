#ifndef __CIFAR_H__
#define __CIFAR_H__

#include <iostream>
#include "data_factory.h"

using namespace std;

class Cifar : public DataFactory {

public:
	Cifar(boost::property_tree::ptree *confReader, string section);
	~Cifar();

	int m_inputDim;
	int m_outputDim;

	int getNumberOfData();
	int getDataSize();
	int getLabelSize();

	void getDataBatch(float* label, float* data, int* indices, int num);
};

#endif