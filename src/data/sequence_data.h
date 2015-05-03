#ifndef _SEQUENCE_DATA_H__
#define _SEQUENCE_DATA_H__

#include "data_factory.h"

using namespace std;

class SequenceData : public DataFactory {
	
private:
	int m_inputSeqLen;
	int m_outputSeqLen;

	int m_inputDim;
	int m_outputDim;

public:
	SequenceData(boost::property_tree::ptree *confReader, string section);
	~SequenceData();

	int getNumberOfData();
	int getDataSize();
	int getLabelSize();
	
	void getDataBatch(float* label, float* data, int* indices, int num);
};

#endif