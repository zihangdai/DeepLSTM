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

    // float data[10] = {1.f, 2.f, 2.f, 4.f, 3.f, 6.f, 4.f, 8.f, 5.f, 10.f};
    // float label[10] = {18.f, 9.f, 16.f, 8.f, 14.f, 7.f, 12.f, 6.f, 10.f, 5.f};    

    int inputSeqLen = confReader->get<int>(section + "max_sequence_length");
    int dimIn = confReader->get<int>(section + "num_neuron_layer_0");
    int dimOut = confReader->get<int>(section + "num_neuron_layer_3");

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

    // printf("LSTM output\n");
    // for (int i=1; i<inputSeqLen+1; ++i) {
    // 	int numNeuron = net->m_vecLayers[1]->m_numNeuron;
    // 	for (int j=0; j<numNeuron; ++j) {
    // 		printf("(%d,%d):%f\n", i, j, net->m_vecLayers[1]->m_outputActs[i][j]);
    // 	}
    // }

    // printf("Softmax output\n");
    // for (int i=1; i<inputSeqLen+1; ++i) {
    // 	int numNeuron = net->m_vecLayers[2]->m_numNeuron;
    // 	for (int j=0; j<numNeuron; ++j) {
    // 		printf("(%d,%d):%f\n", i, j, net->m_vecLayers[2]->m_outputActs[i][j]);
    // 	}
    // }

    // printf("LSTM outputError\n");
    // for (int i=1; i<inputSeqLen+1; ++i) {
    // 	int numNeuron = net->m_vecLayers[1]->m_numNeuron;
    // 	for (int j=0; j<numNeuron; ++j) {
    // 		printf("(%d,%d):%f\n", i, j, net->m_vecLayers[1]->m_outputErrs[i][j]);
    // 	}
    // }

    // printf("LSTM inputError\n");
    // for (int i=1; i<inputSeqLen+1; ++i) {
    // 	int numNeuron = net->m_vecLayers[1]->m_numNeuron;
    // 	for (int j=0; j<numNeuron; ++j) {
    // 		printf("(%d,%d):%f\n", i, j, net->m_vecLayers[1]->m_inputErrs[i][j]);
    // 	}
    // }

    delete confReader;
    DLOG(ERROR) << "delete confReader" << endl;
    delete net;
    DLOG(ERROR) << "delete net" << endl;
    delete [] params;
    delete [] grad;
    DLOG(ERROR) << "delete [] params and delete [] grad" << endl;
    delete [] data;
    delete [] label;
    DLOG(ERROR) << "delete [] data and delete [] label" << endl;
}
