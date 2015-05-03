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
	// ForwardNN   *m_fnn; //TODO	

	float *m_W;
	float *m_gradW;
	float *m_bias;
	float *m_gradBias;	

	int m_paramSize;
	
	int m_dataSize;
	int m_targetSize;

	string m_taskType;

	float *m_outputBuf;
	float *m_outputDelta;

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label, int minibatchSize);
	
	void initParams (float *params);

	void bindWeights(float *params, float *grad);
};


#endif