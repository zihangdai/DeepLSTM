#ifndef __INPUT_LAYER_H__
#define __INPUT_LAYER_H__

#include "layer.h"

using namespace std;

class InputLayer: public RecurrentLayer
{
public:
	InputLayer(int numNeuron, int maxSeqLen);
	~InputLayer();

	/* data */

	/* method */
	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);	
};

#endif