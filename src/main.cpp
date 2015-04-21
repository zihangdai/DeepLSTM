#include "common.h"
#include "lstm_rnn.h"

using namespace std;

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    
    openblas_set_num_threads(1);
    
    boost::property_tree::ptree *confReader = new boost::property_tree::ptree();
    boost::property_tree::ini_parser::read_ini("./config.conf", *confReader);
    
    string section = "LSTM.";
    int max_openmp_threads = confReader->get<int>(section + "max_threads");
    omp_set_num_threads(max_openmp_threads);

    RecurrentNN *net = new LSTM_RNN(confReader, section);

    int paramSize = net->m_nParamSize;
    printf("paramSize:%d\n", paramSize);
    float *params = new float[paramSize];
    float *grad = new float[paramSize];
    net->initParams(params);    

    int inputSeqLen = confReader->get<int>(section + "max_sequence_length");
    int dimIn = confReader->get<int>(section + "num_neuron_layer_0");
    int numlayer = confReader->get<int>(section + "num_layer");
    stringstream ss;
    ss << (numlayer-1);
    int dimOut = confReader->get<int>(section + "num_neuron_layer_" + ss.str());

    float *data = new float[dimIn * inputSeqLen];
    float *label = new float[dimOut * inputSeqLen];
    for (int i=0; i<inputSeqLen; ++i) {
        for (int j=0; j<dimIn; ++j) {
            data[i*dimIn+j] = i*dimIn+j;
        }
        for (int j=0; j<dimOut; ++j) {
            label[i*dimOut+j] = dimOut * inputSeqLen - (i*dimIn+j);
        }
    }       

    float error = net->computeGrad(grad, params, data, label, 1);

    for (int i=0; i<20; ++i) {
        printf("%f\t", grad[i]);
    }
    printf("\n");

    DLOG(ERROR) << "Error: " << error << endl;

    delete confReader;
    delete net;
    delete [] params;
    delete [] grad;
    delete [] data;
    delete [] label;
}
