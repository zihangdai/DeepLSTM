#ifndef __SOFTMAX_LAYER_H__
#define __SOFTMAX_LAYER_H__

#include "layer.h"

using namespace std;

class SoftmaxLayer: public RecurrentLayer
{
public:
	SoftmaxLayer(int numNeuron, int maxSeqLen) : RecurrentLayer(numNeuron, maxSeqLen, numNeuron){};
	~SoftmaxLayer() {};

	/* data */

	/* method */
	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);
};

#endif