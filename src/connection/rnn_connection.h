#ifndef __RECURRENT_CONNECTION_H__
#define __RECURRENT_CONNECTION_H__

#include "common.h"
#include "matrix.h"
#include "rnn_layer.h"

using namespace std;

/****************************************************************
* Recurrent Connection
****************************************************************/

class RecurrentConnection
{
public:
	RecurrentConnection(RecurrentLayer *preLayer, RecurrentLayer *postLayer);
	virtual ~RecurrentConnection() {};

	/* data */
	int m_paramSize;

	float *m_weights;
	float *m_bias;

	float *m_gradWeights;
	float *m_gradBias;
	
	RecurrentLayer *m_preLayer;
	RecurrentLayer *m_postLayer;

	/* method */	
	void virtual feedForward(int inputSeqLen) {};
	void virtual feedBackward(int inputSeqLen) {};

	void virtual forwardStep(int seqIdx) {};
	void virtual backwardStep(int seqIdx) {};

	void virtual forwardStep(int preIdx, int postIdx) {};
	void virtual backwardStep(int preIdx, int postIdx) {};

	void virtual initParams(float *params) {};

	void virtual bindWeights(float *params) {};
	void virtual bindGrads(float *grad) {};
};

/****************************************************************
* Recurrent Full-Connection
****************************************************************/

class RNNFullConnection: public RecurrentConnection
{
public:
	RNNFullConnection(RecurrentLayer *preLayer, RecurrentLayer *postLayer);
	~RNNFullConnection() {};

	/* data */

	/* method */
	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);

	void forwardStep(int seqIdx);
	void backwardStep(int seqIdx);

	void forwardStep(int preIdx, int postIdx);
	void backwardStep(int preIdx, int postIdx);

	void initParams(float *params);

	void bindWeights(float *params);
	void bindGrads(float *grad);
};

/****************************************************************
* Recurrent LSTM-Connection
****************************************************************/

class RNNLSTMConnection: public RecurrentConnection
{
public:
	RNNLSTMConnection(RecurrentLayer *preLayer, RecurrentLayer *postLayer);
	~RNNLSTMConnection() {};

	/* data */

	/* method */
	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);

	void forwardStep(int seqIdx);
	void backwardStep(int seqIdx);

	void forwardStep(int preIdx, int postIdx);
	void backwardStep(int preIdx, int postIdx);	
};

#endif