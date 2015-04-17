#ifndef __INPUT_LAYER_H__
#define __INPUT_LAYER_H__

#include "layer.h"

/****************************************************************
* Input Layer
****************************************************************/
class InputLayer : public RecurrentLayer
{
public:
	InputLayer(int numNeuron, int maxSeqLen) : RecurrentLayer(numNeuron, maxSeqLen, numNeuron) {
		#ifdef DEBUG_LAYER
		printf("InputLayer constructor.\n");
		#endif
	};
	~InputLayer() {};

	/* data */
	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);
};

#endif