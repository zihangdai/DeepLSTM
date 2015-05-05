#ifndef __RNN_EMBEDDING_LAYER_H__
#define __RNN_EMBEDDING_LAYER_H__

#include "rnn_layer.h"

using namespace std;

class EmbeddingLayer: public RecurrentLayer
{
public:
	EmbeddingLayer(int numNeuron, int maxSeqLen, int inputSize);
	~EmbeddingLayer();

	/* data */	

	// embedding matrix
	float *embedMat;

	/* method */
	void initParams(float *params);

	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);
	void resetStates(int inputSeqLen);

	void bindWeights(float *params);
	void bindGrads(float *grad);
	
};

#endif