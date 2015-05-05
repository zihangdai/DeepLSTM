#include "recurrent_forward_nn.h"
#include "rnn_lstm.h"

using namespace std;

RecurrentForwardNN::RecurrentForwardNN(boost::property_tree::ptree *confReader, string section) {
	m_taskType = confReader->get<string>(section + "task_type");

	m_dataSize = confReader->get<int>(section + "data_size");
	m_targetSize = confReader->get<int>(section + "target_size");

	m_rnn = new RNNLSTM(confReader, "RNN.");
	m_outputLayer = new RNNSoftmaxLayer(m_targetSize, 1);
	m_interConnect = new RNNFullConnection(m_rnn->m_vecLayers[m_rnn->m_numLayer-1], m_outputLayer);

	m_paramSize = m_rnn->m_paramSize; 				// m_rnn
	m_paramSize += m_outputLayer->m_paramSize;	// m_outputLayer
	m_paramSize += m_interConnect->m_paramSize;		// m_interConnect 	
}

RecurrentForwardNN::~RecurrentForwardNN() {
	if (m_rnn != NULL) delete m_rnn;
	if (m_outputLayer != NULL) delete [] m_outputLayer;
	if (m_interConnect != NULL) delete [] m_interConnect;
}


float RecurrentForwardNN::computeGrad (float *grad, float *params, float *data, float *target, int minibatchSize) {
	float error = 0.f;
	float corrCount = 0.f;
	
	memset(grad, 0x00, sizeof(float)*m_paramSize);
	bindWeights(params);
	bindGrads(grad);

	float *sampleData = data;
	float *sampleTarget = target;

	RecurrentLayer *rnnInputLayer  = m_rnn->m_vecLayers[0];
	RecurrentLayer *rnnOutputLayer = m_rnn->m_vecLayers[m_rnn->m_numLayer-1];
	
	for (int dataIdx=0; dataIdx<minibatchSize; ++dataIdx) {
		// reset state
		m_rnn->resetStates(m_rnn->m_maxSeqLen);
		m_outputLayer->resetStates(1);		
		
		// ********* forward pass ********* //
		// step 1: set input 
		m_rnn->setInput(sampleData, 1, 1);
		// step 2: rnn feed forward
		m_rnn->feedForward(m_rnn->m_maxSeqLen);
		// step 3: inter connection
		m_interConnect->forwardStep(m_rnn->m_maxSeqLen, 1);
		// step 4: output layer
		m_outputLayer->forwardStep(1);

		// printf("m_outputLayer->m_outputActs[1]:\n");
		// for (int i=0; i<m_targetSize; ++i) {
		// 	printf("%f\t", m_outputLayer->m_outputActs[1][i]);
		// }
		// printf("\n");
		
		// backward pass
		memcpy(m_outputLayer->m_outputErrs[1], sampleTarget, sizeof(float)*m_targetSize);
		
		// compute error		
		int corrIdx = argmax(sampleTarget, m_targetSize);
		int predIdx = argmax(m_outputLayer->m_outputActs[1], m_targetSize);
		// printf("%d vs %d\n", corrIdx, predIdx);

		if (predIdx == corrIdx) {
			corrCount ++;
		}
		error += log(m_outputLayer->m_outputActs[1][corrIdx]);

		// ********* backward pass ********* //
		// step 1: output layer
		m_outputLayer->backwardStep(1);
		// step 2: inter connection
		// for (int i=0; i<m_targetSize; ++i) {
		// 	printf("%f\t", m_rnn->m_vecLayers[m_rnn->m_numLayer-1]->m_outputActs[m_rnn->m_maxSeqLen][i]);
		// }
		// printf("\n");
		m_interConnect->backwardStep(m_rnn->m_maxSeqLen, 1);
		// step 3: rnn feed forward
		m_rnn->feedBackward(m_rnn->m_maxSeqLen);


		// move cursor to new position
		sampleData += m_dataSize;
		sampleTarget += m_targetSize;
	}

	// normalization by number of input sequences and clip gradients to [-1, 1]
	float normFactor = 1.f / (float) minibatchSize;
	for (int dim=0; dim<m_paramSize; ++dim) {
		grad[dim] *= normFactor;
		// if (dim < 20)
		// 	printf("grad[%d]:%f\t", dim, grad[dim]);
		if (grad[dim] < -1.f) {
			grad[dim] = -1.f;
		} else if (grad[dim] > 1.f) {
			grad[dim] = 1.f;
		}
	}
	// printf("\n");
	error *= normFactor;

	printf("Correct rate: %d/%d=%f\n", int(corrCount), minibatchSize, corrCount / float(minibatchSize));

	return error;
}
	
void RecurrentForwardNN::initParams (float *params) {
	// m_rnn
	float *paramsCursor = params;
	m_rnn->initParams(paramsCursor);	

	// m_outputLayer
	paramsCursor += m_rnn->m_paramSize;
	m_outputLayer->initParams(paramsCursor);
	
	// m_interConnect
	paramsCursor += m_outputLayer->m_paramSize;
	m_interConnect->initParams(paramsCursor);
}

void RecurrentForwardNN::bindWeights(float *params) {
	// m_rnn
	float *paramsCursor = params;
	m_rnn->bindWeights(paramsCursor);
	
	// m_outputLayer
	paramsCursor += m_rnn->m_paramSize;
	m_outputLayer->bindWeights(paramsCursor);
	
	// m_interConnect
	paramsCursor += m_outputLayer->m_paramSize;
	m_interConnect->bindWeights(paramsCursor);
}

void RecurrentForwardNN::bindGrads(float *grad) {
	// grad m_rnn
	float *gradCursor = grad;
	m_rnn->bindGrads(gradCursor);

	// grad m_outputLayer
	gradCursor += m_rnn->m_paramSize;
	m_outputLayer->bindGrads(gradCursor);
	
	// grad m_interConnect
	gradCursor += m_outputLayer->m_paramSize;
	m_interConnect->bindGrads(gradCursor);
}