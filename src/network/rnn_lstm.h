#ifndef __RNNLSTM_H__
#define __RNNLSTM_H__

#include "recurrent_nn.h"

using namespace std;

class RNNLSTM: public RecurrentNN
{
public:
	RNNLSTM(boost::property_tree::ptree *confReader, string section);
	~RNNLSTM();

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
	RecurrentConnection *initConnection(int connIdx);	
};

#endif