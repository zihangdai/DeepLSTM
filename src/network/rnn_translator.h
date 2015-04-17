#ifndef __RNN_TRANSLATOR_H__
#define __RNN_TRANSLATOR_H__

#include "lstm_rnn.h"
#include "confreader.h"

class RNNTranslator
{
public:
	RNNTranslator(ConfReader *confReader);
	~RNNTranslator();

	/* data */
	int m_nParamSize;

	float *m_encodingW;
	float *m_decodingW;

	LSTM_RNN *m_encoder;
	LSTM_RNN *m_decoder;	

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label, int minibatchSize);
	void initParams (float *params);

private:
	
};

#endif