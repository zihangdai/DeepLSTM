#ifndef __RNN_TRANSLATOR_H__
#define __RNN_TRANSLATOR_H__

#include "rnn_lstm.h"
#include "common.h"

using namespace std;

class RNNTranslator
{
public:
	RNNTranslator(boost::property_tree::ptree *confReader, string section);
	~RNNTranslator();

	/* data */
	int m_paramSize;
	int m_reverseEncoder;	

	RNNLSTM *m_encoder;
	RNNLSTM *m_decoder;	

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label, int minibatchSize);
	void initParams (float *params);

private:
	void bindWeights(float *params, float *grad);	
};

#endif