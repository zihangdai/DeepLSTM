#include "cifar.h"

using namespace std;

Cifar::Cifar(boost::property_tree::ptree *confReader, string section) {
	// Read Conf
	string fileName = confReader->get<string>(section + "cifar_data_file");

	m_numSample = confReader->get<int>(section + "cifar_sample_num");  

	m_inputDim = confReader->get<int>(section + "cifar_input_dim");
	m_outputDim = confReader->get<int>(section + "cifar_output_dim");

	m_input = new float[m_numSample * m_inputDim];
	m_output = new float[m_numSample * m_outputDim];
	memset(m_output, 0x00, sizeof(float) * m_numSample * m_outputDim);

	// Load Data
	ifstream dataFile(fileName.c_str(), ios::binary);

	// Read real data
	int temp;
	uint8_t temp8;
	for (int i=0; i<m_numSample; i++) {
		dataFile.read((char *)(&temp8), sizeof(char));
		temp = temp8;
		m_output[i * m_outputDim + temp] = 1;
		for(int j=0; j<m_inputDim; j++) {
			dataFile.read((char *)(&temp8), sizeof(char));
			temp = temp8;
			m_input[i * m_inputDim + j] = float(temp) / 255.f;
		}
	}
	dataFile.close();
}

Cifar::~Cifar() {

}

int Cifar::getNumberOfData() {
	return m_numSample;
}

int Cifar::getDataSize() {
	return m_inputDim;
}

int Cifar::getLabelSize() {
	return m_outputDim;
}

void Cifar::getDataBatch(float* label, float* data, int* indices, int num) {
	for (int i=0; i<num; i++) {
		int index = indices[i];
		memcpy(data + i * m_inputDim, m_input + index * m_inputDim, sizeof(float) * m_inputDim);
		memcpy(label + i * m_outputDim, m_output + index * m_outputDim, sizeof(float) * m_outputDim);
	}	
}