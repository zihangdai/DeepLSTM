// #define NDEBUG

#include <time.h>
#include "rnn_translator.h"
#include "sgd.h"

#include <algorithm>
#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);

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
    float *grad = new float[paramSize];
    translator->initParams(params);

    // init sgd optimizer 
    sgdBase *optimizer = new adagrad(confReader, section, paramSize);

    int dataSeqLen = confReader->get<int>(section + "data_sequence_length");
    int targetSeqLen = confReader->get<int>(section + "target_sequence_length");
    int dimIn = confReader->get<int>(section + "data_size");
    int dimOut = confReader->get<int>(section + "target_size");
    int sampleNum = confReader->get<int>(section + "sample_num");

    float *data = new float[dimIn * dataSeqLen * sampleNum];
    float *target = new float[dimOut * targetSeqLen * sampleNum];

    string dataFilename = confReader->get<string>(section + "data_filename");
    string targetFilename = confReader->get<string>(section + "target_filename");

    ifstream datafile (dataFilename, ios::in|ios::binary);
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

    ifstream targetfile (targetFilename, ios::in|ios::binary);
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
    
    int *indices = new int[sampleNum];
    for (int i=0; i<sampleNum; ++i) {
        indices[i] = i;
    }

    // double startTime = CycleTimer::currentSeconds();
    int maxiter = confReader->get<int>(section + "max_iteration");
    int iter = 0, index;
    while (iter < maxiter) {
        random_shuffle(indices, indices+sampleNum);
        for (int i=0; i<sampleNum; ++i) {
            index = indices[i];
            double gradBegTime = CycleTimer::currentSeconds();
            float error = translator->computeGrad(grad, params, data + index * dimIn * dataSeqLen, target + index * dimOut * targetSeqLen, 1);
            double gradEndTime = CycleTimer::currentSeconds();
            cout << "translator computeGrad time: " << gradBegTime - gradEndTime << endl;

            double optBegTime = CycleTimer::currentSeconds();
            optimizer->updateParams(params, grad);
            double optEndTime = CycleTimer::currentSeconds();
            cout << "optimizer updateParams time: " << optBegTime - optEndTime << endl; 

            cout << "Iteration: " << iter << ", Error: " << error << endl;
            iter ++;
        }
    }
    // double endTime = CycleTimer::currentSeconds();
    // printf("Time for %d iterations with %d threads: %f\n", maxiter, max_openmp_threads, endTime - startTime);
    
    return 0;
}