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
	// for (int i=0; i<m_numNeuron; ++i) {
	// 	printf("%f=%f-%f\t", m_inputErrs[seqIdx][i], m_outputActs[seqIdx][i], m_outputErrs[seqIdx][i]);
	// }
	// printf("\n");
}