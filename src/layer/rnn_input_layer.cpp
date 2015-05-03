#include "rnn_input_layer.h"

using namespace std;

// #define DEBUG_INPUT_LAYER

void RNNInputLayer::feedForward(int inputSeqLen) {
	double startTime = CycleTimer::currentSeconds();
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {		
		memcpy(m_outputActs[seqIdx], m_inputActs[seqIdx], sizeof(float) * m_numNeuron);
	}
	double endTime = CycleTimer::currentSeconds();
	#ifdef TIME_SPEED
	printf("RNNInputLayer feedForward time: %f\n", endTime - startTime);
	#endif
}

void RNNInputLayer::feedBackward(int inputSeqLen) {
	#ifdef DEBUG_INPUT_LAYER
	printf("RNNInputLayer feedBackward.\n");
	#endif
}