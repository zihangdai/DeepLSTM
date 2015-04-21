#ifndef __REC_CONNECTION_H__
#define __REC_CONNECTION_H__

#include "common.h"
#include "matrix.h"
#include "layer.h"

using namespace std;

/****************************************************************
* Recurrent Connection
****************************************************************/

class RecConnection
{
public:
	RecConnection(RecurrentLayer *preLayer, RecurrentLayer *postLayer);
	virtual ~RecConnection() {};

	/* data */
	int m_nParamSize;

	float *m_weights;
	float *m_bias;

	float *m_gradWeights;
	float *m_gradBias;
	
	RecurrentLayer *m_preLayer;
	RecurrentLayer *m_postLayer;

	/* method */	
	void virtual feedForward(int inputSeqLen) {};
	void virtual feedBackward(int inputSeqLen) {};

	void virtual initParams(float *params) {};
	void virtual bindWeights(float *params, float *grad) {};
};

/****************************************************************
* Recurrent Full-Connection
****************************************************************/

class FullConnection: public RecConnection
{
public:
	FullConnection(RecurrentLayer *preLayer, RecurrentLayer *postLayer);
	~FullConnection() {};

	/* data */

	/* method */
	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);

	void initParams(float *params);
	void bindWeights(float *params, float *grad);
};

/****************************************************************
* Recurrent LSTM-Connection
****************************************************************/

class LSTMConnection: public RecConnection
{
public:
	LSTMConnection(RecurrentLayer *preLayer, RecurrentLayer *postLayer);
	~LSTMConnection() {};

	/* data */

	/* method */
	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);

	void virtual initParams(float *params) {};
	void virtual bindWeights(float *params, float *grad) {};
};

/****************************************************************
* Forward Connection
****************************************************************/

#endif