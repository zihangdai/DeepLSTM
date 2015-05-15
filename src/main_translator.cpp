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
    float *grad = new float[paramSize];
    translator->initParams(params);

    // init sgd optimizer 
    sgdBase *optimizer = new adagrad(confReader, section, paramSize);

    int dataSeqLen = confReader->get<int>(section + "data_sequence_length");
    int targetSeqLen = confReader->get<int>(section + "target_sequence_length");
    int dimIn = confReader->get<int>(section + "data_size");
    int dimOut = confReader->get<int>(section + "target_size");
    int sampleNum = confReader->get<int>(section + "sample_num");
    int batchSize = confReader->get<int>(section + "batch_size");

    float *data = new float[dimIn * dataSeqLen * sampleNum];
    float *target = new float[dimOut * targetSeqLen * sampleNum];

    float *dataBatch = new float[dimIn * dataSeqLen * batchSize];
    float *targetBatch = new float[dimOut * targetSeqLen * batchSize];

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
    
    int *indices = new int[sampleNum];
    for (int i=0; i<sampleNum; ++i) {
        indices[i] = i;
    }    
    random_shuffle(indices, indices+sampleNum);
    int index = 0;

    double startTime = CycleTimer::currentSeconds();
    int maxiter = confReader->get<int>(section + "max_iteration");
    int iter = 0;
    while (iter < maxiter) {
        if (index + batchSize >= sampleNum) {
            random_shuffle(indices, indices+sampleNum);
            index = 0;
        }
        for (int i=0; i<batchSize; ++i) {
            memcpy(dataBatch + i * dimIn * dataSeqLen, 
                data + indices[index] * dimIn * dataSeqLen, 
                sizeof(float) * dimIn * dataSeqLen);
            memcpy(targetBatch + i * dimOut * targetSeqLen, 
                target + indices[index] * dimOut * targetSeqLen, 
                sizeof(float) * dimOut * targetSeqLen);
            index ++;
        }
        
        double gradBegTime = CycleTimer::currentSeconds();
        float error = translator->computeGrad(grad, params, dataBatch, targetBatch, batchSize);
        double gradEndTime = CycleTimer::currentSeconds();
        cout << "translator computeGrad time: " << gradBegTime - gradEndTime << endl;

        double optBegTime = CycleTimer::currentSeconds();
        optimizer->updateParams(params, grad);
        double optEndTime = CycleTimer::currentSeconds();
        cout << "optimizer updateParams time: " << optBegTime - optEndTime << endl; 

        cout << "Iteration: " << iter << ", Error: " << error << endl;
        iter ++;
    }
    double endTime = CycleTimer::currentSeconds();
    printf("Time for %d iterations with %d threads: %f\n", maxiter, max_openmp_threads, endTime - startTime);

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

    string saveFilename = confReader->get<string>(section + "save_filename");
    ofstream savefile (saveFilename.c_str(), ios::out|ios::binary);
    if (savefile.is_open()) {
        savefile.write ((char *)params, sizeof(float) * paramSize);
        savefile.close();
    } else {
        printf("Failed to open savefile\n");
        exit(1);
    }

    delete [] data;
    delete [] target;

    delete [] dataBatch;
    delete [] targetBatch;
    
    delete [] indices;

    return 0;
}