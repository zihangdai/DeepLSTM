#include "rnn_connection.h"

using namespace std;

// #define DEBUG_CONNECTION

RecurrentConnection::RecurrentConnection(RecurrentLayer *preLayer, RecurrentLayer *postLayer) {
	#ifdef DEBUG_CONNECTION
	printf("RecurrentConnection constructor (%d, %d).\n", preLayer->m_numNeuron, postLayer->m_numNeuron);
	#endif
	m_preLayer = preLayer;
	m_postLayer = postLayer;
}

/****************************************************************
* Recurrent Full-Connection
****************************************************************/

RNNFullConnection::RNNFullConnection(RecurrentLayer *preLayer, RecurrentLayer *postLayer) : RecurrentConnection(preLayer, postLayer) {	
	// weights
	m_paramSize = m_postLayer->m_numNeuron * m_preLayer->m_numNeuron;
	// bias
	m_paramSize += m_postLayer->m_numNeuron;
	#ifdef DEBUG_CONNECTION
	printf("RNNFullConnection constructor %d.\n", m_paramSize);
	#endif
}

void RNNFullConnection::initParams(float *params) {	
	float multiplier = 0.08;
	for (int i=0; i<m_paramSize; ++i) {
		params[i] = multiplier * SYM_UNIFORM_RAND;
		// params[i] = 0.0003;
	}
}

void RNNFullConnection::bindWeights(float *params, float *grad) {
	float *paramsCursor = params;
	float *gradCursor = grad;
	// weights
	m_weights = paramsCursor;
	m_gradWeights = gradCursor;
	paramsCursor += m_postLayer->m_numNeuron * m_preLayer->m_numNeuron;
	gradCursor += m_postLayer->m_numNeuron * m_preLayer->m_numNeuron;
	// bias
	m_bias = paramsCursor;
	m_gradBias = gradCursor;
}

void RNNFullConnection::feedForward(int inputSeqLen) {
	int preNumNeuron = m_preLayer->m_numNeuron;
	int postNumNeuron = m_postLayer->m_numNeuron;	
	double startTime = CycleTimer::currentSeconds();
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {		
		// m_weights
		float *weights = m_weights;
		dot(m_postLayer->m_inputActs[seqIdx], weights, postNumNeuron, preNumNeuron, m_preLayer->m_outputActs[seqIdx], preNumNeuron, 1);
		// m_bias
		elem_accum(m_postLayer->m_inputActs[seqIdx], m_bias, postNumNeuron);
	}
	double endTime = CycleTimer::currentSeconds();
	#ifdef TIME_SPEED
	printf("RNNFullConnection feedForward time: %f\n", endTime - startTime);
	#endif
}

void RNNFullConnection::feedBackward(int inputSeqLen) {
	int preNumNeuron = m_preLayer->m_numNeuron;
	int postNumNeuron = m_postLayer->m_numNeuron;
	double startTime = CycleTimer::currentSeconds();
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		// m_preLayer->m_outputErrs
		trans_dot(m_preLayer->m_outputErrs[seqIdx], m_weights, postNumNeuron, preNumNeuron, m_postLayer->m_inputErrs[seqIdx], postNumNeuron, 1);
		// m_gradWeights
		outer(m_gradWeights, m_postLayer->m_inputErrs[seqIdx], postNumNeuron, m_preLayer->m_outputActs[seqIdx], preNumNeuron);
		// m_gradBias
		elem_accum(m_gradBias, m_postLayer->m_inputErrs[seqIdx], postNumNeuron);
	}
	double endTime = CycleTimer::currentSeconds();
	#ifdef TIME_SPEED
	printf("RNNFullConnection feedBackward time: %f\n", endTime - startTime);
	#endif
}


/****************************************************************
* Recurrent LSTM-Connection
****************************************************************/

RNNLSTMConnection::RNNLSTMConnection(RecurrentLayer *preLayer, RecurrentLayer *postLayer) : RecurrentConnection(preLayer, postLayer) {
	#ifdef DEBUG_CONNECTION
	printf("RNNLSTMConnection constructor.\n");
	#endif
	m_paramSize = 0;
}

void RNNLSTMConnection::feedForward(int inputSeqLen) {
	// independent loop -> use OpenMP potentially
	int inputSize = m_preLayer->m_numNeuron;
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		float *preOutActs = m_preLayer->m_outputActs[seqIdx];
		float *postInActs = m_postLayer->m_inputActs[seqIdx];
		memcpy(postInActs, preOutActs, sizeof(float)*inputSize);
	}
}

void RNNLSTMConnection::feedBackward(int inputSeqLen) {
	// independent loop -> use OpenMP potentially
	int errorSize = m_preLayer->m_numNeuron;
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		float *preOutErrs = m_preLayer->m_outputErrs[seqIdx];
		float *postInErrs = m_postLayer->m_inputErrs[seqIdx];
		memcpy(preOutErrs, postInErrs, sizeof(float)*errorSize);
	}
}