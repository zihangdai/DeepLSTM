#include <time.h>
#include "confreader.h"
#include "rnn_translator.h"
#include "sgd.h"
#include "glog/logging.h"

using namespace std;

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "Test Glog" << endl;
    LOG(WARNING) << "Test Glog" << endl;
    LOG(ERROR) << "Test Glog" << endl;

    srand (time(NULL));
    openblas_set_num_threads(1);    

    ConfReader *confReader = new ConfReader("translator.conf", "Translator");
    int max_openmp_threads = confReader->getInt("max_threads");
    omp_set_num_threads(max_openmp_threads);

    RNNTranslator *translator = new RNNTranslator(confReader);

    int paramSize = translator->m_nParamSize;
    printf("paramSize:%d\n", paramSize);
    float *params = new float[paramSize];
    float *grad = new float[paramSize];
    translator->initParams(params);

    // init sgd optimizer 
    sgdBase *optimizer = new adagrad(confReader, paramSize);
    // sgdBase *optimizer = new adadelta(confReader, paramSize);    

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
            data[i*dimIn+j] = j;
        }
    }
    for (int i=0; i<targetSeqLen; ++i) {
        for (int j=0; j<dimOut; ++j) {
            label[i*dimOut+j] = 2 * j;
        }
    }   
    
    double startTime = CycleTimer::currentSeconds();
    int maxiter = confReader->getInt("max_iteration");
    for (int i=0; i<maxiter; i++) {
        float error = translator->computeGrad(grad, params, data, label, 1);
        optimizer->updateParams(params, grad);
        printf("Error[%d]: %f\n", i, error);
    }
    double endTime = CycleTimer::currentSeconds();
    printf("Time for %d iterations with %d threads: %f\n", maxiter, max_openmp_threads, endTime - startTime);
    
    return 0;
}