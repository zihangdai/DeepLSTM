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
	
    openblas_set_num_threads(1);
	int batchSize = confReader->get<int>(section + "training_batch_size");
	int maxiter = confReader->get<int>(section + "max_iteration");
	int max_openmp_threads = confReader->get<int>(section + "max_threads");
	omp_set_num_threads(max_openmp_threads);	

    printf("openmp threads: %d, %d\n", omp_get_max_threads(), max_openmp_threads);

	// Step 2: Initialize SGD Solver
	section = "SGD.";
	sgdBase *sgdSolver = initSgdSolver(confReader, section, paramSize);

	// step 3: Init Data and allocate related memorys
	section = "Data.";
	DataFactory *dataset = new Mnist(confReader, section);
	int numSample = dataset->getNumberOfData();
	int dataSize  = dataset->getDataSize();
	int labelSize = dataset->getLabelSize();

	float *data  = new float[batchSize * dataSize];
	float *label = new float[batchSize * labelSize];

	int *indices = new int[numSample];
	int *pickIndices = new int[batchSize];
	for (int i=0;i<numSample;i++){
		indices[i]=i;
	}
	std::random_shuffle(indices, indices + numSample);

	// step 4: training
    int iter = 0, index = 0;
    while (iter < maxiter) {
    	// get minibatch data
    	if (index + batchSize >= numSample){
			std::random_shuffle(indices, indices + numSample);
			index = 0;
		}
		for(int i=0;i<batchSize;i++){
			pickIndices[i] = indices[index];
			index++;
		}
		dataset->getDataBatch(label, data, pickIndices, batchSize);
        
        // compute grad
        double gradBegTime = CycleTimer::currentSeconds();
        float error = rnnfnn->computeGrad(grad, params, data, label, batchSize);
        double gradEndTime = CycleTimer::currentSeconds();
        cout << "ComputeGrad time: " << gradBegTime - gradEndTime << endl;

        // update params
        double optBegTime = CycleTimer::currentSeconds();
        sgdSolver->updateParams(params, grad);
        double optEndTime = CycleTimer::currentSeconds();
        cout << "UpdateParams time: " << optBegTime - optEndTime << endl; 

        cout << "Iteration: " << iter << ", Error: " << error << endl;
        iter ++;
    }

	return 0;
}