#ifndef __MSE_LAYER_H__
#define __MSE_LAYER_H__

#include "layer.h"

using namespace std;

class MSELayer: public RecurrentLayer
{
public:
	MSELayer(int numNeuron, int maxSeqLen) : RecurrentLayer(numNeuron, maxSeqLen, numNeuron){};
	~MSELayer() {};

	/* data */

	/* method */
	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);
};

#endif