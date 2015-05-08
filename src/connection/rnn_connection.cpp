#include "rnn_connection.h"

using namespace std;

RecurrentConnection::RecurrentConnection(RecurrentLayer *preLayer, RecurrentLayer *postLayer) {
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
}

void RNNFullConnection::initParams(float *params) {	
	float multiplier = 0.08;
	for (int i=0; i<m_paramSize; ++i) {
		params[i] = multiplier * SYM_UNIFORM_RAND;
		// params[i] = 0.0003;
	}
}

void RNNFullConnection::bindWeights(float *params) {
	// weights
	float *paramsCursor = params;
	m_weights = paramsCursor;
	
	// bias
	paramsCursor += m_postLayer->m_numNeuron * m_preLayer->m_numNeuron;	
	m_bias = paramsCursor;
}

void RNNFullConnection::bindGrads(float *grad) {
	// gradWeights
	float *gradCursor = grad;
	m_gradWeights = gradCursor;

	// gradBias
	gradCursor += m_postLayer->m_numNeuron * m_preLayer->m_numNeuron;
	m_gradBias = gradCursor;
}

void RNNFullConnection::feedForward(int inputSeqLen) {
	int preNumNeuron = m_preLayer->m_numNeuron;
	int postNumNeuron = m_postLayer->m_numNeuron;
	double startTime = CycleTimer::currentSeconds();
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		// m_weights
		dot(m_postLayer->m_inputActs[seqIdx], m_weights, postNumNeuron, preNumNeuron, m_preLayer->m_outputActs[seqIdx], preNumNeuron, 1);
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

void RNNFullConnection::forwardStep(int seqIdx) {
	int preNumNeuron = m_preLayer->m_numNeuron;
	int postNumNeuron = m_postLayer->m_numNeuron;
	// m_postLayer->m_inputActs[seqIdx] += m_weights * m_preLayer->m_outputActs[seqIdx]
	dot(m_postLayer->m_inputActs[seqIdx], m_weights, postNumNeuron, preNumNeuron, m_preLayer->m_outputActs[seqIdx], preNumNeuron, 1);
	// m_postLayer->m_inputActs[seqIdx] += m_bias
	elem_accum(m_postLayer->m_inputActs[seqIdx], m_bias, postNumNeuron);
}

void RNNFullConnection::backwardStep(int seqIdx) {
	int preNumNeuron = m_preLayer->m_numNeuron;
	int postNumNeuron = m_postLayer->m_numNeuron;
	// m_preLayer->m_outputErrs[seqIdx] += m_weights^T * m_postLayer->m_inputErrs[seqIdx]
	trans_dot(m_preLayer->m_outputErrs[seqIdx], m_weights, postNumNeuron, preNumNeuron, m_postLayer->m_inputErrs[seqIdx], postNumNeuron, 1);
	// m_gradWeights += m_postLayer->m_inputErrs[seqIdx] * m_preLayer->m_outputActs[seqIdx]^T
	outer(m_gradWeights, m_postLayer->m_inputErrs[seqIdx], postNumNeuron, m_preLayer->m_outputActs[seqIdx], preNumNeuron);
	// m_gradBias += m_postLayer->m_inputErrs[seqIdx]
	elem_accum(m_gradBias, m_postLayer->m_inputErrs[seqIdx], postNumNeuron);
}

void RNNFullConnection::forwardStep(int preIdx, int postIdx) {
	int preNumNeuron = m_preLayer->m_numNeuron;
	int postNumNeuron = m_postLayer->m_numNeuron;
	// m_postLayer->m_inputActs[postIdx] += m_weights * m_preLayer->m_outputActs[preIdx]
	dot(m_postLayer->m_inputActs[postIdx], m_weights, postNumNeuron, preNumNeuron, m_preLayer->m_outputActs[preIdx], preNumNeuron, 1);
	// m_postLayer->m_inputActs[postIdx] += m_bias
	elem_accum(m_postLayer->m_inputActs[postIdx], m_bias, postNumNeuron);
}

void RNNFullConnection::backwardStep(int preIdx, int postIdx) {
	int preNumNeuron = m_preLayer->m_numNeuron;
	int postNumNeuron = m_postLayer->m_numNeuron;
	// m_preLayer->m_outputErrs[preIdx] += m_weights^T * m_postLayer->m_inputErrs[postIdx]
	trans_dot(m_preLayer->m_outputErrs[preIdx], m_weights, postNumNeuron, preNumNeuron, m_postLayer->m_inputErrs[postIdx], postNumNeuron, 1);
	// m_gradWeights += m_postLayer->m_inputErrs[postIdx] * m_preLayer->m_outputActs[preIdx]^T
	outer(m_gradWeights, m_postLayer->m_inputErrs[postIdx], postNumNeuron, m_preLayer->m_outputActs[preIdx], preNumNeuron);
	// m_gradBias += m_postLayer->m_inputErrs[postIdx]
	elem_accum(m_gradBias, m_postLayer->m_inputErrs[postIdx], postNumNeuron);
}

/****************************************************************
* Recurrent LSTM-Connection
****************************************************************/

RNNLSTMConnection::RNNLSTMConnection(RecurrentLayer *preLayer, RecurrentLayer *postLayer) : RecurrentConnection(preLayer, postLayer) {
	m_paramSize = 0;
}

void RNNLSTMConnection::feedForward(int inputSeqLen) {
	int inputSize = m_preLayer->m_numNeuron;
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		float *preOutActs = m_preLayer->m_outputActs[seqIdx];
		float *postInActs = m_postLayer->m_inputActs[seqIdx];
		memcpy(postInActs, preOutActs, sizeof(float)*inputSize);
	}
}

void RNNLSTMConnection::feedBackward(int inputSeqLen) {
	int errorSize = m_preLayer->m_numNeuron;
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		float *preOutErrs = m_preLayer->m_outputErrs[seqIdx];
		float *postInErrs = m_postLayer->m_inputErrs[seqIdx];
		memcpy(preOutErrs, postInErrs, sizeof(float)*errorSize);
	}
}

void RNNLSTMConnection::forwardStep(int seqIdx) {
	// m_preLayer -> m_postLayer
	memcpy(m_postLayer->m_inputActs[seqIdx], m_preLayer->m_outputActs[seqIdx], sizeof(float)*m_preLayer->m_numNeuron);
}

void RNNLSTMConnection::backwardStep(int seqIdx) {
	// m_postLayer -> m_preLayer
	memcpy(m_preLayer->m_outputErrs[seqIdx], m_postLayer->m_inputErrs[seqIdx], sizeof(float)*m_preLayer->m_numNeuron);
}

void RNNLSTMConnection::forwardStep(int preIdx, int postIdx) {
	// m_preLayer -> m_postLayer
	memcpy(m_postLayer->m_inputActs[postIdx], m_preLayer->m_outputActs[preIdx], sizeof(float)*m_preLayer->m_numNeuron);
}

void RNNLSTMConnection::backwardStep(int preIdx, int postIdx) {
	// m_postLayer -> m_preLayer
	memcpy(m_preLayer->m_outputErrs[preIdx], m_postLayer->m_inputErrs[postIdx], sizeof(float)*m_preLayer->m_numNeuron);
}