#include "recurrent_forward_nn.h"
#include "rnn_lstm.h"

using namespace std;

RecurrentForwardNN::RecurrentForwardNN(boost::property_tree::ptree *confReader, string section) {
	m_taskType = confReader->get<string>(section + "task_type");

	m_dataSize = confReader->get<int>(section + "data_size");
	m_targetSize = confReader->get<int>(section + "target_size");

	m_outputBuf = new float[m_targetSize];
	m_outputDelta = new float[m_targetSize];

	m_rnn = new RNNLSTM(confReader, "RNN.");

	m_paramSize = m_rnn->m_paramSize; // m_rnn
	// m_paramSize += m_targetSize * m_rnn->m_outputSize; // m_W
	// m_paramSize += m_targetSize; // m_bias

}

RecurrentForwardNN::~RecurrentForwardNN() {
	if (m_outputBuf != NULL) delete [] m_outputBuf;
	if (m_outputDelta != NULL) delete [] m_outputDelta;
	if (m_rnn != NULL) delete m_rnn;
}


float RecurrentForwardNN::computeGrad (float *grad, float *params, float *data, float *target, int minibatchSize) {
	float error = 0.f;
	float corrCount = 0.f;
	
	memset(grad, 0x00, sizeof(float)*m_paramSize);
	bindWeights(params, grad);

	float *sampleData = data;
	float *sampleTarget = target;

	RecurrentLayer *rnnInputLayer  = m_rnn->m_vecLayers[0];
	RecurrentLayer *rnnOutputLayer = m_rnn->m_vecLayers[m_rnn->m_numLayer-1];
	
	for (int dataIdx=0; dataIdx<minibatchSize; ++dataIdx) {
		// reset state
		m_rnn->resetStates(m_rnn->m_maxSeqLen);
		memset(m_outputBuf, 0x00, sizeof(float)*m_targetSize);

		// forward pass
		for (int seqIdx=1; seqIdx<=m_rnn->m_maxSeqLen; ++seqIdx) {
			memcpy(rnnInputLayer->m_inputActs[seqIdx], sampleData, sizeof(float)*m_dataSize);
		}
		m_rnn->feedForward(m_rnn->m_maxSeqLen);
		
		// dot(m_outputBuf, m_W, m_targetSize, m_rnn->m_outputSize, rnnOutputLayer->m_outputActs[m_rnn->m_maxSeqLen], m_rnn->m_outputSize, 1);
		// elem_accum(m_outputBuf, m_bias, m_targetSize);
		// if (m_taskType == "classification") {
		// 	softmax(m_outputBuf, m_outputBuf, m_targetSize);
		// }

		// memset(m_outputDelta, 0x00, sizeof(float) * m_targetSize);
		// elem_sub(m_outputDelta, m_outputBuf, sampleTarget, m_targetSize);
		
		// // compute error
		// float maxProb = 0.f;
		// int corrIdx = -1, predIdx = -1;
		// for (int i=0; i<m_targetSize; ++i) {			
		// 	if (m_taskType == "classification") {
		// 		error += sampleTarget[i] * log(m_outputBuf[i]);				
		// 		if (sampleTarget[i] == 1) {
		// 			corrIdx = i;
		// 		}
		// 		if (maxProb < m_outputBuf[i]) {
		// 			maxProb = m_outputBuf[i];
		// 			predIdx = i;
		// 		}
		// 	} else if (m_taskType == "regression") {
		// 		error += m_outputDelta[i] * m_outputDelta[i];
		// 	}
		// }
		// if (predIdx == corrIdx) {
		// 	corrCount ++;
		// }

		// // backward pass
		// outer(m_gradW, m_outputDelta, m_targetSize, rnnOutputLayer->m_outputActs[m_rnn->m_maxSeqLen], m_rnn->m_outputSize);
		// elem_accum(m_gradBias, m_outputDelta, m_targetSize);

		// trans_dot(rnnOutputLayer->m_outputErrs[m_rnn->m_maxSeqLen], m_W, m_targetSize, m_rnn->m_outputSize, m_outputDelta, m_targetSize, 1);

		for (int seqIdx=1; seqIdx<=m_rnn->m_maxSeqLen; ++seqIdx) {
			memcpy(rnnOutputLayer->m_outputErrs[seqIdx], sampleTarget, sizeof(float)*m_targetSize);
		}

		for (int seqIdx=1; seqIdx<=m_rnn->m_maxSeqLen; ++seqIdx) {
			for (int i=0; i<m_rnn->m_outputSize; ++i) {				
				error += sampleTarget[i] * log(rnnOutputLayer->m_outputActs[seqIdx][i]);
			}			
		}
		m_rnn->feedBackward(m_rnn->m_maxSeqLen);

		// move cursor to new position
		sampleData += m_dataSize;
		sampleTarget += m_targetSize;
	}

	// normalization by number of input sequences and clip gradients to [-1, 1]
	float normFactor = 1.f / (float) minibatchSize;
	for (int dim=0; dim<m_paramSize; ++dim) {
		grad[dim] *= normFactor;
		if (grad[dim] < -1.f) {
			grad[dim] = -1.f;
		} else if (grad[dim] > 1.f) {
			grad[dim] = 1.f;
		}
	}
	error *= normFactor;

	printf("Correct rate: %d/%d=%f\n", int(corrCount), minibatchSize, corrCount / float(minibatchSize));

	return error;
}
	
void RecurrentForwardNN::initParams (float *params) {
	float *paramsCursor = params;

	m_rnn->initParams(paramsCursor);
	paramsCursor += m_rnn->m_paramSize;

	float multiplier = 0.08;
	for (int i=0; i<m_targetSize * (m_rnn->m_outputSize+1); ++i) {
		paramsCursor[i] = multiplier * SYM_UNIFORM_RAND;
	}
}

void RecurrentForwardNN::bindWeights(float *params, float *grad) {
	float *paramsCursor = params;
	float *gradCursor = grad;
	
	// m_rnn
	m_rnn->bindWeights(paramsCursor, gradCursor);
	paramsCursor += m_rnn->m_paramSize;
	gradCursor += m_rnn->m_paramSize;

	// // Weights
	// m_W = paramsCursor;
	// m_gradW = gradCursor;

	// paramsCursor += m_targetSize * m_rnn->m_outputSize;
	// gradCursor += m_targetSize * m_rnn->m_outputSize;

	// // bias
	// m_bias = paramsCursor;
	// m_gradBias = gradCursor;
}