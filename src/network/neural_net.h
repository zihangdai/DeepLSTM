#ifndef __NEURAL_NET_H__
#define __NEURAL_NET_H__

#include <stdlib.h>
#include <string.h>
#include "confreader.h"
#include "input_layer.h"
#include "lstm_layer.h"
#include "softmax_layer.h"
#include "mse_layer.h"
#include "connection.h"

using namespace std;

class RecurrentNN
{
public:
	RecurrentNN() {};
	virtual ~RecurrentNN() {};

	/* data */
	int m_numLayer;
	int m_maxSeqLen;
	int m_nParamSize;

	string m_errorType;

	int *m_numNeuronList;
	string *m_layerTypeList;
	string *m_connTypeList;

	vector<RecurrentLayer *> m_vecLayers;
	vector<RecConnection *> m_vecConnections;

	/* method */
	float virtual computeGrad (float *grad, float *params, float *data, float *label, int minibatchSize) {return 0.f;};
	void virtual initParams (float *params) {};
};

#endif