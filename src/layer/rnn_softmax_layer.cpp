#include "rnn_softmax_layer.h"

void RNNSoftmaxLayer::feedForward(int inputSeqLen) {
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		softmax(m_outputActs[seqIdx], m_inputActs[seqIdx], m_numNeuron);
	}
}

void RNNSoftmaxLayer::feedBackward(int inputSeqLen) {
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		elem_sub(m_inputErrs[seqIdx], m_outputActs[seqIdx], m_outputErrs[seqIdx], m_numNeuron);
	}
}

void RNNSoftmaxLayer::forwardStep(int seqIdx) {
	softmax(m_outputActs[seqIdx], m_inputActs[seqIdx], m_numNeuron);
}

void RNNSoftmaxLayer::backwardStep(int seqIdx) {
	elem_sub(m_inputErrs[seqIdx], m_outputActs[seqIdx], m_outputErrs[seqIdx], m_numNeuron);
}