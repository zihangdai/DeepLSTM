#include "common.h"
#include "recurrent_forward_nn.h"
#include "sgd.h"
#include "mnist.h"
#include "data_factory.h"

using namespace std;

sgdBase * initSgdSolver (boost::property_tree::ptree *confReader, string section, int paramSize) {
	int solverType = confReader->get<int>(section+"solver_type");
	sgdBase *sgdSolver;
	switch (solverType) {
		// sgdBasic
		case 0: {
			printf("Init basic sgd solver.\n");
			sgdSolver = new sgdBasic(confReader, section, paramSize);
			break;
		}
		// adagrad
		case 1: {
			printf("Init adagrad solver.\n");
			sgdSolver = new adagrad(confReader, section, paramSize);
			break;
		}
		// adadelta
		case 2: {
			sgdSolver = new adadelta(confReader, section, paramSize);
			printf("Init adadelta solver.\n");
			break;
		}
		// rmsprop
		case 3: {
			sgdSolver = new rmsprop(confReader, section, paramSize);
			printf("Init rmsprop solver.\n");
			break;
		}
		default: {
			printf("Error solver type.\n");
			exit(-1);
		}
	}
	return sgdSolver;
}

int main(int argc, char const *argv[]) {
	
	srand(time(NULL));
	
	// step 0: Init conf and read basic slave conf
	boost::property_tree::ptree *confReader = new boost::property_tree::ptree();
	boost::property_tree::ini_parser::read_ini("rnnfnn.conf", *confReader);

	// step 1: Init Model and allocate related memorys
	string section = "RNNFNN.";
	RecurrentForwardNN *rnnfnn = new RecurrentForwardNN(confReader, section);

	int paramSize = rnnfnn->m_paramSize;
	printf("paramSize: %d\n", paramSize);
	
	float *params = new float[paramSize];
	float *grad   = new float[paramSize];
	rnnfnn->initParams(params);
	
	openblas_set_num_threads(1);
	int batchSize = confReader->get<int>(section + "training_batch_size");
	int maxiter = confReader->get<int>(section + "max_iteration");
	int max_openmp_threads = confReader->get<int>(section + "max_threads");
	omp_set_num_threads(max_openmp_threads);

	printf("openmp threads: %d, %d\n", omp_get_max_threads(), max_openmp_threads);

	// Step 2: Initialize SGD Solver
	section = "SGD.";
	sgdBase *sgdSolver = initSgdSolver(confReader, section, paramSize);

	// step 3: Init Training Data and allocate related memorys
	section = "Data.";
	DataFactory *trainData = new Mnist(confReader, section);
	int numSample = trainData->getNumberOfData();
	int dataSize  = trainData->getDataSize();
	int labelSize = trainData->getLabelSize();
	printf("Finish load data\n");

	float *data  = new float[batchSize * dataSize];
	float *label = new float[batchSize * labelSize];

	int *indices = new int[numSample];
	int *pickIndices = new int[batchSize];
	for (int i=0;i<numSample;i++){
		indices[i]=i;
	}
	std::random_shuffle(indices, indices + numSample);

	// step 4: Init validation Data
    section = "ValidData.";
    DataFactory *validData = new Mnist(confReader, section);

    // step 5: training
	int iter = 0, index = 0, epoch = 0;
	while (iter < maxiter) {
		// get minibatch data
		if (index + batchSize > numSample){
            printf("Cross Validation After Epoch %d\n", epoch++);
            rnnfnn->predict(params, validData->m_input, validData->m_output, validData->m_numSample);
			std::random_shuffle(indices, indices + numSample);
			index = 0;
		}
		for(int i=0;i<batchSize;i++){
			pickIndices[i] = indices[index];
			index++;
		}
		trainData->getDataBatch(label, data, pickIndices, batchSize);
		
		// compute grad
		double gradBegTime = CycleTimer::currentSeconds();
		float error = rnnfnn->computeGrad(grad, params, data, label, batchSize);
		double gradEndTime = CycleTimer::currentSeconds();
		cout << "ComputeGrad time: " << gradEndTime - gradBegTime << endl;

		// update params
		double optBegTime = CycleTimer::currentSeconds();
		sgdSolver->updateParams(params, grad);
		double optEndTime = CycleTimer::currentSeconds();
		cout << "UpdateParams time: " << optEndTime - optBegTime << endl; 

		cout << "Iteration: " << iter << ", Error: " << error << endl;
		iter ++;
	}

	// step 6: save trained weights
	section = "RNNFNN.";
	string saveFilename = confReader->get<string>(section + "save_filename");
	ofstream savefile (saveFilename.c_str(), ios::out|ios::binary);
	if (savefile.is_open()) {
		savefile.write ((char *)params, sizeof(float) * paramSize);
		savefile.close();
	} else {
		printf("Failed to open savefile\n");
		exit(1);
	}

	// step 7: delete allocated memory
	delete [] params;
	delete [] grad;

	delete [] data;
	delete [] label;

	delete [] indices;
	delete [] pickIndices;

	delete rnnfnn;
	delete confReader;
	delete sgdSolver;
	delete trainData;
	delete validData;

	return 0;
}