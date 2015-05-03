#ifndef __RNN_SOFTMAX_LAYER_H__
#define __RNN_SOFTMAX_LAYER_H__

#include "rnn_layer.h"

using namespace std;

class RNNSoftmaxLayer: public RecurrentLayer
{
public:
	RNNSoftmaxLayer(int numNeuron, int maxSeqLen) : RecurrentLayer(numNeuron, maxSeqLen, numNeuron){};
	~RNNSoftmaxLayer() {};

	/* data */

	/* method */
	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);
};

#endif