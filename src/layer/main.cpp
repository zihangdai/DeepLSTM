#include "layer.h"
#include "lstm_layer.h"

int main () {
    RecurrentLayer * L = new LSTMLayer(2,2,2);
    L->reshape(5);
    int paramSize = L->m_nParamSize;
    float *params = new float [paramSize];
    float *grad = new float [paramSize];
    L->initParams(params);
    for (int i=0; i<paramSize; i++) {
    	printf("%f\t", params[i]);
    }
    printf("\n");
    L->bindWeights(params, grad);
    L->resetStates(5);
    int inputSeqLen = 4;
    for (int seqIdx=1; seqIdx<inputSeqLen+1; ++seqIdx) {
        for (int i=0; i<L->m_inputSize; i++) {
            L->m_inputActs[seqIdx][i] = float(seqIdx);
        }
    }
    L->feedForward(inputSeqLen);
    printf("m_outputActs\n");
    for (int seqIdx=0; seqIdx<inputSeqLen+2; ++seqIdx) {
        for (int i=0; i<L->m_numNeuron; i++) {
            printf("%f\t", L->m_outputActs[seqIdx][i]);
        }
        printf("\n");
    }
    for (int seqIdx=1; seqIdx<inputSeqLen+1; ++seqIdx) {
        for (int i=0; i<L->m_numNeuron; i++) {
            L->m_outputErrs[seqIdx][i] = float(10-seqIdx);
        }
    }
    memset(grad, 0x00, sizeof(float)*paramSize);
    L->feedBackward(inputSeqLen);
    printf("m_inputErrs\n");
    for (int seqIdx=0; seqIdx<inputSeqLen+2; ++seqIdx) {
        for (int i=0; i<L->m_numNeuron; i++) {
            printf("%f\t", L->m_inputErrs[seqIdx][i]);
        }
        printf("\n");
    }
    printf("grad\n");
    for (int i=0; i<paramSize; i++) {
        printf("%f\t", grad[i]);
    }
    printf("\n");
    printf("grad\n");
    for (int i=0; i<paramSize; i++) {
        printf("%f\t", grad[paramSize-1-i]);
    }
    printf("\n");
    
    delete L;
    delete [] params;
    delete [] grad;
    return 0;
}
