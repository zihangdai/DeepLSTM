#ifndef __FNN_LAYER_H__
#define __FNN_LAYER_H__

#include "common.h"
#include "matrix.h"
#include "nonlinearity.h"

using namespace std;

/****************************************************************
* Forward Layer
****************************************************************/
class ForwardLayer
{
public:
	/* constructor */
	ForwardLayer(int numNeuron);

	// destructor */
	virtual ~ForwardLayer();

	/* data */

	// structure parameters
	int m_numNeuron;
	int m_paramSize;	

	// internal states
	float *m_inputActs;
	float *m_outputActs;

	float *m_inputErrs;
	float *m_outputErrs;
	
	/* method */
	void virtual feedForward() {};
	void virtual feedBackward() {};
};

/****************************************************************
* Forward Softmax Layer
****************************************************************/

class FNNSoftmaxLayer: public ForwardLayer
{
public:
	FNNSoftmaxLayer (int numNeuron) : ForwardLayer(numNeuron) {};
	~FNNSoftmaxLayer ();

	/* data */

	/* method */
	void feedForward();
	void feedBackward();
};

/****************************************************************
* Sigmoid Layer
****************************************************************/

// class FNNSigmoidLayer: public ForwardLayer
// {
// public:
// 	FNNSigmoidLayer (int m_numNeuron);
// 	~FNNSigmoidLayer ();

// 	/* data */

// 	 method 

// 	void feedForward();
// 	void feedBackward();
// };