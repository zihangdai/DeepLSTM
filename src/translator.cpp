// #define NDEBUG

#include <time.h>
#include "rnn_translator.h"
#include "sgd.h"

using namespace std;

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);

    // srand (time(NULL));
    openblas_set_num_threads(1);    

    boost::property_tree::ptree *confReader = new boost::property_tree::ptree();
    boost::property_tree::ini_parser::read_ini("translator.conf", *confReader);
    string section = "Translator.";

    int max_openmp_threads = confReader->get<int>(section + "max_threads");
    omp_set_num_threads(max_openmp_threads);

    RNNTranslator *translator = new RNNTranslator(confReader, section);

    int paramSize = translator->m_nParamSize;
    printf("paramSize:%d\n", paramSize);
    float *params = new float[paramSize];
    float *grad = new float[paramSize];
    translator->initParams(params);

    // init sgd optimizer 
    sgdBase *optimizer = new adagrad(confReader, section, paramSize);
    // sgdBase *optimizer = new adadelta(confReader, paramSize);    

    // float data[10] = {1.f, 2.f, 2.f, 4.f, 3.f, 6.f, 4.f, 8.f, 5.f, 10.f};
    // float label[10] = {18.f, 9.f, 16.f, 8.f, 14.f, 7.f, 12.f, 6.f, 10.f, 5.f};

    int dataSeqLen = confReader->get<int>(section + "data_sequence_length");
    int targetSeqLen = confReader->get<int>(section + "target_sequence_length");
    int dimIn = confReader->get<int>(section + "data_size");
    int dimOut = confReader->get<int>(section + "target_size");

    float *data = new float[dimIn * dataSeqLen];
    float *label = new float[dimOut * targetSeqLen];
    for (int i=0; i<dataSeqLen; ++i) {
        for (int j=0; j<dimIn; ++j) {
            data[i*dimIn+j] = i;
        }
    }
    for (int i=0; i<targetSeqLen; ++i) {
        for (int j=0; j<dimOut; ++j) {
            label[i*dimOut+j] = 2 * i;
        }
    }   
    
    double startTime = CycleTimer::currentSeconds();
    int maxiter = confReader->get<int>("max_iteration");
    for (int i=0; i<maxiter; i++) {
        double gradBegTime = CycleTimer::currentSeconds();
        float error = translator->computeGrad(grad, params, data, label, 1);
        double gradEndTime = CycleTimer::currentSeconds();
        DLOG(ERROR) << "translator computeGrad time: " << gradBegTime - gradEndTime << endl;

        double optBegTime = CycleTimer::currentSeconds();
        optimizer->updateParams(params, grad);
        double optEndTime = CycleTimer::currentSeconds();
        DLOG(ERROR) << "optimizer updateParams time: " << optBegTime - optEndTime << endl; 

        DLOG(ERROR) << "Error: " << error << endl;
    }
    double endTime = CycleTimer::currentSeconds();
    printf("Time for %d iterations with %d threads: %f\n", maxiter, max_openmp_threads, endTime - startTime);
    
    return 0;
}