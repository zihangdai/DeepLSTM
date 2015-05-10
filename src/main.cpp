#include "common.h"
#include "rnn_lstm.h"
#include "sgd.h"
#include "mnist.h"
#include "data_factory.h"

using namespace std;

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    
    // step 0: Init conf and read basic slave conf
    boost::property_tree::ptree *confReader = new boost::property_tree::ptree();
    boost::property_tree::ini_parser::read_ini("./config.conf", *confReader);
    
    // step 1: Init Model and allocate related memorys
    string section = "LSTM.";
    RecurrentNN *net = new RNNLSTM(confReader, section);

    int paramSize = net->m_paramSize;
    printf("paramSize:%d\n", paramSize);
    float *params = new float[paramSize];
    float *grad = new float[paramSize];
    net->initParams(params);

    openblas_set_num_threads(1);
    int batchSize = confReader->get<int>(section + "training_batch_size");
    int maxiter = confReader->get<int>(section + "max_iteration");
    int max_openmp_threads = confReader->get<int>(section + "max_threads");
    omp_set_num_threads(max_openmp_threads);
    omp_set_nested(0);

    printf("openmp threads: max threads %d, nested %d\n", omp_get_max_threads(), omp_get_nested());

    // Step 2: Initialize SGD Solver
    section = "SGD.";
    sgdBase *sgdSolver = new adagrad(confReader, section, paramSize);

    // step 3: Init Training Data and allocate related memorys
    section = "Data.";
    DataFactory *trainData = new Mnist(confReader, section);
    int numSample = trainData->getNumberOfData();
    int dataSize  = trainData->getDataSize();
    int labelSize = trainData->getLabelSize();

    float *data  = new float[batchSize * dataSize];
    float *label = new float[batchSize * labelSize];

    int *indices = new int[numSample];
    int *pickIndices = new int[batchSize];
    for (int i=0;i<numSample;i++){
        indices[i]=i;
    }
    std::random_shuffle(indices, indices + numSample);

    // step 4: training
    int iter = 0, index = 0, epoch = 0;
    while (iter < maxiter) {
        // get minibatch data
        if (index + batchSize > numSample){            
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
        float error = net->computeGrad(grad, params, data, label, batchSize);
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
        
    delete [] params;
    delete [] grad;

    delete [] data;
    delete [] label;

    delete [] indices;
    delete [] pickIndices;

    delete confReader;
    delete net;
    delete trainData;
    delete sgdSolver;
}
