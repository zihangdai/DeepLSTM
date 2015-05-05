#ifndef __RECURRENT_FORWARD_NEURAL_NETWORK_H__
#define __RECURRENT_FORWARD_NEURAL_NETWORK_H__

#include "common.h"
#include "matrix.h"
#include "nonlinearity.h"

#include "recurrent_nn.h"
#include "rnn_lstm.h"
// #include "forward_nn.h" // TODO

using namespace std;

class RecurrentForwardNN
{
public:
	RecurrentForwardNN(boost::property_tree::ptree *confReader, string section);
	~RecurrentForwardNN();

	/* data */
	RNNLSTM *m_rnn;
	
	RNNFullConnection *m_interConnect;

	RecurrentLayer *m_outputLayer;

	int m_paramSize;
	
	int m_dataSize;
	int m_targetSize;

	string m_taskType;

	/* method */
	float predict (float *params, float *data, float *target, int batchSize);
	float computeGrad (float *grad, float *params, float *data, float *label, int minibatchSize);
	
	void initParams (float *params);

	void bindWeights(float *params);
	void bindGrads(float *grad);
};


#endif