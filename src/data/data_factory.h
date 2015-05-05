#ifndef __DATA_FACTORY_H__
#define __DATA_FACTORY_H__

#include <iostream>
#include <fstream>
#include "common.h"

using namespace std;

class DataFactory
{
public:
	DataFactory() {};
	virtual ~DataFactory() {};
	
	virtual int getNumberOfData() {return 0;};
	virtual int getDataSize() {return 0;};
	virtual int getLabelSize() {return 0;};

	virtual void getDataBatch(float* label, float* data, int* indices, int num) {};
	
	void getAllData(float* label, float* data) {
		data = m_input;
		label = m_output;
	};

	int m_numSample;

	float *m_input;
	float *m_output;
};

#endif
