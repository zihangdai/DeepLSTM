#ifndef __LSTM_RNN_H__
#define __LSTM_RNN_H__

#include "neural_net.h"

using namespace std;

class LSTM_RNN: public RecurrentNN
{
public:
	LSTM_RNN();
	~LSTM_RNN();

	/* data */

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label, int minibatchSize);
	void initParams (float *params);

	float computeError(float *sampleTarget, int inputSeqLen);

	void feedBackward(int inputSeqLen);
	void feedForward(int inputSeqLen);

	void bindWeights(float *params, float *grad);
	void resetStates(int inputSeqLen);
	
private:
	RecurrentLayer *initLayer (int layerIdx);
	RecConnection *initConnection(int connIdx);	
};

#endif