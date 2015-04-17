#include "softmax_layer.h"

void SoftmaxLayer::feedForward(int inputSeqLen) {
	double startTime = CycleTimer::currentSeconds();
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		#ifdef DEBUG_SOFTMAX_LAYER
		printf("SoftmaxLayer feedForward from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
		#endif
		softmax(m_outputActs[seqIdx], m_inputActs[seqIdx], m_numNeuron);		
	}
	double endTime = CycleTimer::currentSeconds();
	printf("SoftmaxLayer feedForward time: %f\n", endTime - startTime);
	
}

void SoftmaxLayer::feedBackward(int inputSeqLen) {
	double startTime = CycleTimer::currentSeconds();
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		#ifdef DEBUG_SOFTMAX_LAYER
		printf("SoftmaxLayer feedBackward from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
		#endif
		elem_sub(m_inputErrs[seqIdx], m_outputActs[seqIdx], m_outputErrs[seqIdx], m_numNeuron);		
	}
	double endTime = CycleTimer::currentSeconds();
	printf("SoftmaxLayer feedBackward time: %f\n", endTime - startTime);
}