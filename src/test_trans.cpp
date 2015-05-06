// #define NDEBUG

#include <time.h>
#include "rnn_translator.h"
#include "sgd.h"

#include <algorithm>
#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char* argv[]) {

	srand (time(NULL));
	openblas_set_num_threads(1);

	boost::property_tree::ptree *confReader = new boost::property_tree::ptree();
	boost::property_tree::ini_parser::read_ini("translator.conf", *confReader);
	string section = "Translator.";

	int max_openmp_threads = confReader->get<int>(section + "max_threads");
	omp_set_num_threads(max_openmp_threads);

	RNNTranslator *translator = new RNNTranslator(confReader, section);

	int paramSize = translator->m_paramSize;
	printf("paramSize:%d\n", paramSize);
	float *params = new float[paramSize];

	int dataSeqLen = confReader->get<int>(section + "data_sequence_length");
	int targetSeqLen = confReader->get<int>(section + "target_sequence_length");
	int dimIn = confReader->get<int>(section + "data_size");
	int dimOut = confReader->get<int>(section + "target_size");
	int sampleNum = confReader->get<int>(section + "sample_num");
	int batchSize = confReader->get<int>(section + "batch_size");

	float *data = new float[dimIn * dataSeqLen * sampleNum];
	float *target = new float[dimOut * targetSeqLen * sampleNum];	

	string dataFilename = confReader->get<string>(section + "data_filename");
	string targetFilename = confReader->get<string>(section + "target_filename");

	ifstream datafile (dataFilename.c_str(), ios::in|ios::binary);
	if (datafile.is_open()) {
		datafile.seekg (0, ios::end);
		int size = datafile.tellg();
		if (size != sizeof(float) * dimIn * dataSeqLen * sampleNum) {
			printf("Wrong memory size for data\n");
			datafile.close();
			exit(1);
		}
		datafile.seekg (0, ios::beg);
		datafile.read ((char *)data, size);
		datafile.close();
	} else {
		printf("Failed to open datafile\n");
		exit(1);
	}

	ifstream targetfile (targetFilename.c_str(), ios::in|ios::binary);
	if (targetfile.is_open()) {
		targetfile.seekg (0, ios::end);
		int size = targetfile.tellg();
		if (size != sizeof(float) * dimOut * targetSeqLen * sampleNum) {
			printf("Wrong memory size for target\n");
			targetfile.close();
			exit(1);
		}
		targetfile.seekg (0, ios::beg);
		targetfile.read ((char *)target, size);
		targetfile.close();
	} else {
		printf("Failed to open targetfile\n");
		exit(1);
	}

	string saveFilename = confReader->get<string>(section + "save_filename");
	ifstream savefile (saveFilename.c_str(), ios::out|ios::binary);
	if (savefile.is_open()) {
		savefile.read ((char *)params, sizeof(float) * paramSize);
		savefile.close();
	} else {
		printf("Failed to open savefile\n");
		exit(1);
	}

	float *input = new float [dataSeqLen];
	float *predict = new float [dataSeqLen];
	for (int i=0; i<dataSeqLen; ++i) {
		input[i] = (float) rand() / (float) (RAND_MAX);
	}
	
	translator->translate(params, input, predict, 1);

	sort(input, input+dataSeqLen);

	for (int i=0; i<dataSeqLen; ++i) {
		printf("%f,%f\t", input[i], predict[i]);
	}
	printf("\n");

	delete [] params;

	delete [] data;
	delete [] target;

	return 0;
}