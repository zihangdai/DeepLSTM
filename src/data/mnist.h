#ifndef __MNIST_H__
#define __MNIST_H__

#include <iostream>
#include "data_factory.h"

using namespace std;

class Mnist : public DataFactory {

public:
    Mnist(boost::property_tree::ptree *confReader, string section);
    ~Mnist();

    int m_inputDim;
    int m_outputDim;

    int getNumberOfData();
    int getDataSize();
    int getLabelSize();

    void getDataBatch(float* label, float* data, int* indices, int num);
};

#endif