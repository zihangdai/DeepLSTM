#ifndef __RNN_TRANSLATOR_H__
#define __RNN_TRANSLATOR_H__

#include "lstm_rnn.h"
#include "common.h"

class RNNTranslator
{
public:
	RNNTranslator(ConfReader *confReader);
	~RNNTranslator();

	/* data */
	int m_nParamSize;
	int m_reverseEncoder;

	float *m_encodingW;
	float *m_gradEncodingW;

	LSTM_RNN *m_encoder;
	LSTM_RNN *m_decoder;	

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label, int minibatchSize);
	void initParams (float *params);

private:
	void bindWeights(float *params, float *grad);	
};

#endif