#ifndef __RNN_MSE_LAYER_H__
#define __RNN_MSE_LAYER_H__

#include "rnn_layer.h"

using namespace std;

class RNNMSELayer: public RecurrentLayer
{
public:
	RNNMSELayer(int numNeuron, int maxSeqLen) : RecurrentLayer(numNeuron, maxSeqLen, numNeuron){};
	~RNNMSELayer() {};

	/* data */

	/* method */
	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);
};

#endif