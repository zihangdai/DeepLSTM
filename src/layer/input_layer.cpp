#include "input_layer.h"

using namespace std;

#define DEBUG_INPUT_LAYER 1

void InputLayer::feedForward(int inputSeqLen) {
	double startTime = CycleTimer::currentSeconds();
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {		
		memcpy(m_outputActs[seqIdx], m_inputActs[seqIdx], sizeof(float) * m_numNeuron);
	}
	double endTime = CycleTimer::currentSeconds();
	#ifdef TIME_SPEED
	printf("InputLayer feedForward time: %f\n", endTime - startTime);
	#endif
}

void InputLayer::feedBackward(int inputSeqLen) {
	#ifdef DEBUG_INPUT_LAYER
	printf("InputLayer feedBackward.\n");
	#endif
}