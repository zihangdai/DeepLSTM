#include "sequence_data.h"

using namespace std;

SequenceData::SequenceData(boost::property_tree::ptree *confReader, string section) {
	string inputFile = confReader->get<string>(section + "seqdata_input_file");
	string outputFile = confReader->get<string>(section + "seqdata_output_file");

	m_numSample = confReader->get<int>(section + "seqdata_sample_num");

	m_inputSeqLen = confReader->get<int>(section + "seqdata_input_len");
	m_outputSeqLen = confReader->get<int>(section + "seqdata_output_len");

	m_inputDim = confReader->get<int>(section + "seqdata_input_dim");
	m_outputDim = confReader->get<int>(section + "seqdata_output_dim");

	m_input = new float[m_numSample * m_inputSeqLen * m_inputDim];
	m_output = new float[m_numSample * m_outputSeqLen * m_outputDim];

	ifstream inputfile (inputFile.c_str(), ios::in|ios::binary);
	if (inputfile.is_open()) {
		inputfile.seekg (0, ios::end);
		int size = inputfile.tellg();
		if (size != sizeof(float) * m_numSample * m_inputSeqLen * m_inputDim) {
			printf("Wrong memory size for sequence input\n");
			inputfile.close();
			exit(1);
		}
		inputfile.seekg (0, ios::beg);
		inputfile.read ((char *)m_input, size);
		inputfile.close();
	} else {
		printf("Failed to open inputfile\n");
		exit(1);
	}

	ifstream outputfile (outputFile.c_str(), ios::in|ios::binary);
	if (outputfile.is_open()) {
		outputfile.seekg (0, ios::end);
		int size = outputfile.tellg();
		if (size != sizeof(float) * m_numSample * m_outputSeqLen * m_outputDim) {
			printf("Wrong memory size for sequence output\n");
			outputfile.close();
			exit(1);
		}
		outputfile.seekg (0, ios::beg);
		outputfile.read ((char *)m_output, size);
		outputfile.close();
	} else {
		printf("Failed to open outputfile\n");
		exit(1);
	}
}

SequenceData::~SequenceData() {
	if (m_input != NULL) delete [] m_input;
	if (m_output != NULL) delete [] m_output;
}

int SequenceData::getNumberOfData() {
	return m_numSample;
}

int SequenceData::getDataSize() {
	return m_inputDim * m_inputSeqLen;
}
int SequenceData::getLabelSize() {
	return m_outputDim * m_outputSeqLen;
}

void SequenceData::getDataBatch(float* label, float* data, int* indices, int num) {
	for (int i=0; i<num; ++i) {
		int index = indices[i];
		memcpy(data + i * m_inputSeqLen * m_inputDim, 
			m_input + index * m_inputSeqLen * m_inputDim, 
			sizeof(float) * m_inputSeqLen * m_inputDim);
		memcpy(label + i * m_outputSeqLen * m_outputDim, 
			m_output + index * m_outputSeqLen * m_outputDim, 
			sizeof(float) * m_outputSeqLen * m_outputDim);
	}
}
