#ifndef __NEURAL_NET_H__
#define __NEURAL_NET_H__

#include <stdlib.h>
#include <string.h>
#include "confreader.h"
#include "lstm_layer.h"
#include "softmax_layer.h"
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

	int *m_numNeuronList;
	string *m_layerTypeList;
	string *m_connTypeList;

	vector<RecurrentLayer *> m_vecLayers;
	vector<RecConnection *> m_vecConnections;

	/* method */
	float virtual computeGrad (float *grad, float *params, float *data, float *label, int minibatchSize) {return 0.f;};
	void virtual initParams (float *params) {};
};

class LSTM_RNN: public RecurrentNN
{
public:
	LSTM_RNN(ConfReader *confReader);
	~LSTM_RNN();

	/* data */

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label, int minibatchSize);
	void initParams (float *params);

private:
	RecurrentLayer *initLayer (int layerIdx);
	RecConnection *initConnection(int connIdx);
	void bindWeights(float *params, float *grad);
	void resetStates(int inputSeqLen);
};

#endif