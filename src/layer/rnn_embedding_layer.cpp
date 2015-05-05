#include "rnn_embedding_layer.h"

RNNEmbeddingLayer::RNNEmbeddingLayer(int numNeuron, int maxSeqLen, int vocSize) : RecurrentLayer(numNeuron, maxSeqLen, 1) {
	m_vocSize = vocSize;
	m_embedMat = new float [m_vocSize * m_numNeuron];
	m_paramSize = m_vocSize * m_numNeuron;
}

RNNEmbeddingLayer::~RNNEmbeddingLayer() {
	if (m_embedMat != NULL) delete [] m_embedMat;
}

void RNNEmbeddingLayer::initParams(float *params) {
	float multiplier = 0.01;
	srand(time(NULL));
	for (int i=0; i<m_paramSize; ++i) {
		params[i] = multiplier * SYM_UNIFORM_RAND;
	}
}

void RNNEmbeddingLayer::feedForward(int inputSeqLen) {
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		int vocIdx = (int) (m_inputActs[seqIdx][0] + 0.5f);
		memcpy(m_outputActs[seqIdx], m_embedMat+vocIdx*m_numNeuron, sizeof(float)*m_numNeuron);
	}
}

void RNNEmbeddingLayer::feedBackward(int inputSeqLen) {
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		int vocIdx = (int) (m_inputActs[seqIdx][0] + 0.5f);
		elem_accum(m_gradEmbedMat+vocIdx*m_numNeuron, m_outputErrs, sizeof(float)*m_numNeuron);
	}
}

void RNNEmbeddingLayer::forwardStep(int seqIdx) {
	int vocIdx = (int) (m_inputActs[seqIdx][0] + 0.5f);
	memcpy(m_outputActs[seqIdx], m_embedMat+vocIdx*m_numNeuron, sizeof(float)*m_numNeuron);
}

void RNNEmbeddingLayer::backwardStep(int seqIdx) {
	int vocIdx = (int) (m_inputActs[seqIdx][0] + 0.5f);
	elem_accum(m_gradEmbedMat+vocIdx*m_numNeuron, m_outputErrs, sizeof(float)*m_numNeuron);
}

void RNNEmbeddingLayer::bindWeights(float *params) {
	m_embedMat = params;
}

void RNNEmbeddingLayer::bindGrads(float *grad) {
	m_gradEmbedMat = grad;
}