#include "mnist.h"

using namespace std;

Mnist::Mnist(boost::property_tree::ptree *confReader, string section) {
	// Read Conf
	string inputFile = confReader->get<string>(section + "mnist_input_file");
	string outputFile = confReader->get<string>(section + "mnist_output_file");

	m_numSample = confReader->get<int>(section + "mnist_sample_num");  

	m_inputDim = confReader->get<int>(section + "mnist_input_dim");
	m_outputDim = confReader->get<int>(section + "mnist_output_dim");

	m_input = new float[m_numSample * m_inputDim];
	m_output = new float[m_numSample * m_outputDim];
	memset(m_output, 0x00, sizeof(float) * m_numSample * m_outputDim);
	
	// Load Data		
	ifstream dataFile(inputFile.c_str(), ios::binary);
	ifstream labelFile(outputFile.c_str(), ios::binary);
	
	int temp;
	uint8_t temp8;
	
	// Skip file headers
	for(int i=0; i<2; ++i) {
		labelFile.read((char *)(&temp), sizeof(int));
		dataFile.read((char *)(&temp), sizeof(int));
		dataFile.read((char *)(&temp), sizeof(int));
	}

	// Read real data
	for (int i=0; i<m_numSample; i++) {
		for(int j=0; j<m_inputDim; j++) {
			dataFile.read((char *)(&temp8), sizeof(char));
			temp = temp8;
			m_input[i * m_inputDim + j] = float(temp) / 255.f;
		}
		labelFile.read((char *)(&temp8), sizeof(char));
		temp = temp8;
		m_output[i * m_outputDim + temp] = 1;
	}

	dataFile.close();
	labelFile.close();
}

Mnist::~Mnist() {
	if (m_input != NULL) delete [] m_input;
	if (m_output != NULL) delete [] m_output;
}

int Mnist::getNumberOfData() {
	return m_numSample;
}

int Mnist::getDataSize() {
	return m_inputDim;
}

int Mnist::getLabelSize() {
	return m_outputDim;
}

void Mnist::getDataBatch(float* label, float* data, int* indices, int num) {
	for (int i=0; i<num; i++) {
		int index = indices[i];
		memcpy(data + i * m_inputDim, m_input + index * m_inputDim, sizeof(float) * m_inputDim);
		memcpy(label + i * m_outputDim, m_output + index * m_outputDim, sizeof(float) * m_outputDim);
	}
}