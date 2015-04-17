#ifndef __LSTM_RNN_H__
#define __LSTM_RNN_H__

#include "neural_net.h"

class LSTM_RNN: public RecurrentNN
{
public:
	LSTM_RNN(ConfReader *confReader);
	~LSTM_RNN();

	/* data */

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label, int minibatchSize);
	void initParams (float *params);

	float computeError(float *sampleLabel, int inputSeqLen);
	void feedBackward(float *sampleLabel, int inputSeqLen);
	void feedForward(float *sampleData, int inputSeqLen);

private:
	RecurrentLayer *initLayer (int layerIdx);
	RecConnection *initConnection(int connIdx);
	void bindWeights(float *params, float *grad);
	void resetStates(int inputSeqLen);
};

#endif