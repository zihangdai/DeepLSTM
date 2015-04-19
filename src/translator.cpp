#include "confreader.h"
#include "rnn_translator.h"
#include "sgd.h"

int main() {
    openblas_set_num_threads(1);
    int max_openmp_threads = 20;
    omp_set_num_threads(max_openmp_threads);

    ConfReader *confReader = new ConfReader("translator.conf", "Translator");
    RNNTranslator *translator = new RNNTranslator(confReader);

    int paramSize = translator->m_nParamSize;
    printf("paramSize:%d\n", paramSize);
    float *params = new float[paramSize];
    float *grad = new float[paramSize];
    translator->initParams(params);

    // init sgd optimizer 
    // sgdBase *optimizer = new adagrad(confReader, paramSize);
    sgdBase *optimizer = new adadelta(confReader, paramSize);    

    // float data[10] = {1.f, 2.f, 2.f, 4.f, 3.f, 6.f, 4.f, 8.f, 5.f, 10.f};
    // float label[10] = {18.f, 9.f, 16.f, 8.f, 14.f, 7.f, 12.f, 6.f, 10.f, 5.f};

    int dataSeqLen = confReader->getInt("data_sequence_length");
    int targetSeqLen = confReader->getInt("target_sequence_length");
    int dimIn = confReader->getInt("data_size");
    int dimOut = confReader->getInt("target_size");

    float *data = new float[dimIn * dataSeqLen];
    float *label = new float[dimOut * targetSeqLen];
    for (int i=0; i<dataSeqLen; ++i) {
        for (int j=0; j<dimIn; ++j) {
            data[i*dimIn+j] = 1.f * (float(rand()) / float(RAND_MAX) + 0.5);
        }
    }
    for (int i=0; i<targetSeqLen; ++i) {
        for (int j=0; j<dimOut; ++j) {
            label[i*dimOut+j] = 5.f * (float(rand()) / float(RAND_MAX) + 0.5);
        }
    }   
    
    int maxiter = confReader->getInt("max_iteration");
    for (int i=0; i<maxiter; i++) {
        float error = translator->computeGrad(grad, params, data, label, 1);
        optimizer->updateParams(params, grad);
        printf("Error: %f\n", error);
    }
    
    return 0;
}