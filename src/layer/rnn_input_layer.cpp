#include "rnn_input_layer.h"

using namespace std;

void RNNInputLayer::feedForward(int inputSeqLen) {
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {		
		memcpy(m_outputActs[seqIdx], m_inputActs[seqIdx], sizeof(float) * m_numNeuron);
	}
}

void RNNInputLayer::forwardStep(int seqIdx) {
	memcpy(m_outputActs[seqIdx], m_inputActs[seqIdx], sizeof(float) * m_numNeuron);
}