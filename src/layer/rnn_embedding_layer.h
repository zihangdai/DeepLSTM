#ifndef __RNN_EMBEDDING_LAYER_H__
#define __RNN_EMBEDDING_LAYER_H__

#include "rnn_layer.h"

using namespace std;

class RNNEmbeddingLayer: public RecurrentLayer
{
public:
	RNNEmbeddingLayer(int numNeuron, int maxSeqLen, int inputSize, int vocSize);
	~RNNEmbeddingLayer();

	/* data */	
	int m_vocSize;

	// embedding matrix
	float *m_embedMat;
	float *m_gradEmbedMat;

	/* method */
	void initParams(float *params);

	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);

	void forwardStep(int seqIdx);
	void backwardStep(int seqIdx);

	void bindWeights(float *params);
	void bindGrads(float *grad);
	
};

#endif