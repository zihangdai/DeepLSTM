#ifndef __RNN_INPUT_LAYER_H__
#define __RNN_INPUT_LAYER_H__

#include "rnn_layer.h"

/****************************************************************
* Input Layer
****************************************************************/
class RNNInputLayer : public RecurrentLayer
{
public:
	RNNInputLayer(int numNeuron, int maxSeqLen) : RecurrentLayer(numNeuron, maxSeqLen, numNeuron) {
		#ifdef DEBUG_LAYER
		printf("RNNInputLayer constructor.\n");
		#endif
	};
	~RNNInputLayer() {};

	/* data */
	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);
};

#endif