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
	string m_taskType;

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label, int minibatchSize);
	void initParams (float *params);

	float computeError(float *sampleTarget, int inputSeqLen);
	void getPredict(float *samplePredict, int inputSeqLen);

	void feedBackward(int inputSeqLen);
	void feedForward(int inputSeqLen);

	void forwardStep(int seqIdx);
	void backwardStep(int seqIdx);

	void setInput(float *input, int begIdx, int endIdx, int stride=1);
	void setTarget(float *target, int begIdx, int endIdx, int stride=1);

	void bindWeights(float *params);
	void bindGrads(float *grad);

	void resetStates(int inputSeqLen);		

private:
	RecurrentLayer *initLayer (int layerIdx);
	RecurrentConnection *initConnection(int connIdx);	
};

#endif