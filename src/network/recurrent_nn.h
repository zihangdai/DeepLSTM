#ifndef __RECURRENT_NEURAL_NETWORK_H__
#define __RECURRENT_NEURAL_NETWORK_H__

#include "common.h"

#include "rnn_layer.h"
#include "rnn_input_layer.h"
#include "rnn_lstm_layer.h"
#include "rnn_softmax_layer.h"
#include "rnn_mse_layer.h"

#include "rnn_connection.h"

using namespace std;

class RecurrentNN
{
public:
	RecurrentNN() {};
	virtual ~RecurrentNN() {};

	/* data */
	int m_numLayer;
	int m_maxSeqLen;
	int m_paramSize;

	int m_inputSize;
	int m_outputSize;

	string m_errorType;

	int *m_numNeuronList;
	
	string *m_layerTypeList;
	string *m_connTypeList;

	vector<RecurrentLayer *> m_vecLayers;
	vector<RecurrentConnection *> m_vecConnections;

	/* method */
	float virtual computeGrad (float *grad, float *params, float *data, float *label, int minibatchSize) {return 0.f;};
	void virtual initParams (float *params) {};
};

#endif