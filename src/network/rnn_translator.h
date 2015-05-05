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
	void initParams (float *params);

	float computeGrad (float *grad, float *params, float *input, float *label, int minibatchSize);

	float translate (float *params, float *input, float *predict, int batchSize);

private:
	void bindWeights(float *params);
	void bindGrads(float *grad);
};

#endif