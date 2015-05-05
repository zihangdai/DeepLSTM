#ifndef __RNN_INPUT_LAYER_H__
#define __RNN_INPUT_LAYER_H__

#include "rnn_layer.h"

/****************************************************************
* Input Layer
****************************************************************/
class RNNInputLayer : public RecurrentLayer
{
public:
	RNNInputLayer(int numNeuron, int maxSeqLen) : RecurrentLayer(numNeuron, maxSeqLen, numNeuron) {};
	~RNNInputLayer() {};

	/* data */
	void feedForward(int inputSeqLen);
	// void feedBackward(int inputSeqLen);

	void forwardStep(int seqIdx);
	// void backwardStep(int seqIdx);
};

#endif