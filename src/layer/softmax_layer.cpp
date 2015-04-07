#include "softmax_layer.h"

void SoftmaxLayer::feedForward(int inputSeqLen) {
	printf("SoftmaxLayer feedForward\n");
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		softmax(m_outputActs[seqIdx], m_inputActs[seqIdx], m_numNeuron);
	}
}

void SoftmaxLayer::feedBackward(int inputSeqLen) {
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		elem_sub(m_inputErrs[seqIdx], m_outputActs[seqIdx], m_outputErrs[seqIdx], m_numNeuron);
	}
}