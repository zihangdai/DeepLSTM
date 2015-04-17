#include "mse_layer.h"

void MSELayer::feedForward(int inputSeqLen) {
	double startTime = CycleTimer::currentSeconds();
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {		
		memset(m_outputActs[seqIdx], m_inputActs[seqIdx], sizeof(float) * m_numNeuron);
	}
	double endTime = CycleTimer::currentSeconds();
	printf("MSELayer feedForward time: %f\n", endTime - startTime);
}

void MSELayer::feedBackward(int inputSeqLen) {
	double startTime = CycleTimer::currentSeconds();
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		elem_sub(m_inputErrs[seqIdx], m_outputActs[seqIdx], m_outputErrs[seqIdx], m_numNeuron);
	}
	double endTime = CycleTimer::currentSeconds();
	printf("MSELayer feedBackward time: %f\n", endTime - startTime);
}