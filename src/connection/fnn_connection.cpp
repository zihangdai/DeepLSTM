#include "fnn_connection.h"

using namespace std;

/****************************************************************
* Forward Connection
****************************************************************/

ForwardConnection::ForwardConnection(ForwardLayer *preLayer, ForwardLayer *postLayer) {	
	m_preLayer = preLayer;
	m_postLayer = postLayer;
}

/****************************************************************
* Forward Full-Connection
****************************************************************/

FNNFullConnection::FNNFullConnection(ForwardLayer *preLayer, ForwardLayer *postLayer) : ForwardConnection(preLayer, postLayer) {
	// weights
	m_paramSize = m_postLayer->m_numNeuron * m_preLayer->m_numNeuron;
	// bias
	m_paramSize += m_postLayer->m_numNeuron;
}

void FNNFullConnection::initParams(float *params) {	
	int fanIn = m_preLayer->m_numNeuron;
	int fanOut = m_preLayer->m_numNeuron;
	float multiplier = 4.f * sqrt(6.f / (float)(fanIn + fanOut + 1));
	for (int i=0; i<m_paramSize; ++i) {
		params[i] = multiplier * SYM_UNIFORM_RAND;
	}
}

void FNNFullConnection::bindWeights(float *params) {
	// weights
	float *paramsCursor = params;	
	m_weights = paramsCursor;
	
	// bias
	paramsCursor += m_postLayer->m_numNeuron * m_preLayer->m_numNeuron;		
	m_bias = paramsCursor;
}

void FNNFullConnection::bindGrads(float *grad) {
	// weights
	float *gradCursor = grad;	
	m_gradWeights = gradCursor;
		
	// bias
	gradCursor += m_postLayer->m_numNeuron * m_preLayer->m_numNeuron;
	m_gradBias = gradCursor;
}

void FNNFullConnection::feedForward() {
	int preNumNeuron = m_preLayer->m_numNeuron;
	int postNumNeuron = m_postLayer->m_numNeuron;		
	// m_weights	
	dot(m_postLayer->m_inputActs[seqIdx], m_weights, postNumNeuron, preNumNeuron, m_preLayer->m_outputActs[seqIdx], preNumNeuron, 1);
	// m_bias
	elem_accum(m_postLayer->m_inputActs[seqIdx], m_bias, postNumNeuron);	
}

void FNNFullConnection::feedBackward() {
	int preNumNeuron = m_preLayer->m_numNeuron;
	int postNumNeuron = m_postLayer->m_numNeuron;	
	// m_preLayer->m_outputErrs
	trans_dot(m_preLayer->m_outputErrs[seqIdx], m_weights, postNumNeuron, preNumNeuron, m_postLayer->m_inputErrs[seqIdx], postNumNeuron, 1);
	// m_gradWeights
	outer(m_gradWeights, m_postLayer->m_inputErrs[seqIdx], postNumNeuron, m_preLayer->m_outputActs[seqIdx], preNumNeuron);
	// m_gradBias
	elem_accum(m_gradBias, m_postLayer->m_inputErrs[seqIdx], postNumNeuron);
}