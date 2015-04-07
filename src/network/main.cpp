#include "neural_net.h"
#include "confreader.h"

int main() {
    ConfReader *confReader = new ConfReader("config.conf", "LSTM");
    RecurrentNN *net = new LSTM_RNN(confReader);
    int paramSize = net->m_nParamSize;
    printf("paramSize:%d\n", paramSize);
    float *params = new float[paramSize];
    float *grad = new float[paramSize];
    net->initParams(params);
    printf("params:\n");
    for (int i=0; i<paramSize; ++i) {
    	printf("%f\t", params[i]);
    }
    printf("\n");

    float data[10] = {1.f, 2.f, 2.f, 4.f, 3.f, 6.f, 4.f, 8.f, 5.f, 10.f};
    float label[10] = {18.f, 9.f, 16.f, 8.f, 14.f, 7.f, 12.f, 6.f, 10.f, 5.f};

    int inputSeqLen = 5;
    float error = net->computeGrad(grad, params, data, label);

    printf("Error: %f\n", error);


    printf("LSTM output\n");
    for (int i=1; i<inputSeqLen+1; ++i) {
    	int numNeuron = net->m_vecLayers[1]->m_numNeuron;
    	for (int j=0; j<numNeuron; ++j) {
    		printf("(%d,%d):%f\n", i, j, net->m_vecLayers[1]->m_outputActs[i][j]);
    	}
    }

    printf("Softmax output\n");
    for (int i=1; i<inputSeqLen+1; ++i) {
    	int numNeuron = net->m_vecLayers[2]->m_numNeuron;
    	for (int j=0; j<numNeuron; ++j) {
    		printf("(%d,%d):%f\n", i, j, net->m_vecLayers[2]->m_outputActs[i][j]);
    	}
    }

    printf("LSTM inputError\n");
    for (int i=1; i<inputSeqLen+1; ++i) {
    	int numNeuron = net->m_vecLayers[1]->m_numNeuron;
    	for (int j=0; j<numNeuron; ++j) {
    		printf("(%d,%d):%f\n", i, j, net->m_vecLayers[1]->m_inputErrs[i][j]);
    	}
    }

    delete confReader;
    delete net;
    delete [] params;
    delete [] grad;
}
