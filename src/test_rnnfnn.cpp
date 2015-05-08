#include "common.h"
#include "recurrent_forward_nn.h"
#include "sgd.h"
#include "mnist.h"
#include "data_factory.h"

using namespace std;

int main(int argc, char const *argv[]) {
	// step 0: Init conf and read basic slave conf
	boost::property_tree::ptree *confReader = new boost::property_tree::ptree();
	boost::property_tree::ini_parser::read_ini("rnnfnn.conf", *confReader);

	// step 1: Init Model and allocate related memorys
	string section = "RNNFNN.";
	RecurrentForwardNN *rnnfnn = new RecurrentForwardNN(confReader, section);

	int paramSize = rnnfnn->m_paramSize;
	
	float *params = new float[paramSize];
	string saveFilename = confReader->get<string>(section + "save_filename");
	ifstream savefile (saveFilename.c_str(), ios::out|ios::binary);
	if (savefile.is_open()) {
		savefile.read ((char *)params, sizeof(float) * paramSize);
		savefile.close();
	} else {
		printf("Failed to open savefile\n");
		exit(1);
	}

	// step 2: Init Test Data and allocate related memorys
	section = "TestData.";
	DataFactory *testData = new Mnist(confReader, section);

	// step 4: testing
	float error = rnnfnn->predict(params, testData->m_input, testData->m_output, testData->m_numSample);
	cout << "Test Error: " << error << endl;

	// step 6: delete allocated memory
	delete [] params;

	delete rnnfnn;
	delete testData;
	
	return 0;
}