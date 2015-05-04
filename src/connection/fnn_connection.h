#ifndef __FORWARD_CONNECTION_H__
#define __FORWARD_CONNECTION_H__

#include "common.h"
#include "matrix.h"
#include "FNN_layer.h"

using namespace std;

/****************************************************************
* Forward Connection
****************************************************************/

class ForwardConnection
{
public:
	ForwardConnection(ForwardLayer *preLayer, ForwardLayer *postLayer);
	virtual ~ForwardConnection() {};

	/* data */
	int m_paramSize;

	// trainable weights
	float *m_weights;
	float *m_bias;

	float *m_gradWeights;
	float *m_gradBias;

	ForwardLayer *m_preLayer;
	ForwardLayer *m_postLayer;

	/* method */	
	void virtual feedForward() {};
	void virtual feedBackward() {};

	void virtual initParams(float *params) {};
	void virtual bindWeights(float *params, float *grad) {};
};

/****************************************************************
* Forward Full-Connection
****************************************************************/

class FNNFullConnection: public ForwardConnection
{
public:
	FNNFullConnection(ForwardLayer *preLayer, ForwardLayer *postLayer);
	~FNNFullConnection() {};

	/* data */

	/* method */
	void feedForward();
	void feedBackward();

	void initParams(float *params);
	void bindWeights(float *params, float *grad);
};

#endif