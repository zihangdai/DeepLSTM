#include "rnn_mse_layer.h"

void RNNMSELayer::feedForward(int inputSeqLen) {
	double startTime = CycleTimer::currentSeconds();
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {		
		memcpy(m_outputActs[seqIdx], m_inputActs[seqIdx], sizeof(float) * m_numNeuron);
	}
	double endTime = CycleTimer::currentSeconds();
	#ifdef TIME_SPEED
	printf("RNNMSELayer feedForward time: %f\n", endTime - startTime);
	#endif
}

void RNNMSELayer::feedBackward(int inputSeqLen) {
	double startTime = CycleTimer::currentSeconds();
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		elem_sub(m_inputErrs[seqIdx], m_outputActs[seqIdx], m_outputErrs[seqIdx], m_numNeuron);
	}
	double endTime = CycleTimer::currentSeconds();
	#ifdef TIME_SPEED
	printf("RNNMSELayer feedBackward time: %f\n", endTime - startTime);
	#endif
}

