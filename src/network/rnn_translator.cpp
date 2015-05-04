#include <cmath>
#include <iostream>
#include "rnn_translator.h"

// #define DEBUG_RNN_TRANSLATOR 1

using namespace std;

RNNTranslator::RNNTranslator(boost::property_tree::ptree *confReader, string section) {
	// init encoder and decoder
	m_reverseEncoder = confReader->get<int>(section + "reverse_encoder");

	m_encoder = new RNNLSTM(confReader, "Encoder.");
	m_decoder = new RNNLSTM(confReader, "Decoder.");
	
	// compute paramSize
	m_paramSize = 0;
	m_paramSize += m_encoder->m_paramSize;
	m_paramSize += m_decoder->m_paramSize;	

	#ifdef DEBUG_RNN_TRANSLATOR
	printf("RNNTranslator constructor finished.\n");
	#endif
}

RNNTranslator::~RNNTranslator() {
	if (m_encoder != NULL) {
		delete m_encoder;
	}
	if (m_decoder != NULL) {
		delete m_decoder;
	}
}

void RNNTranslator::initParams (float *params) {
	float *paramsCursor = params;
	
	m_encoder->initParams(paramsCursor);
	
	paramsCursor += m_encoder->m_paramSize;

	m_decoder->initParams(paramsCursor);
}

void RNNTranslator::bindWeights(float *params, float *grad) {
	float *paramsCursor = params;
	float *gradCursor = grad;
	
	m_encoder->bindWeights(paramsCursor, gradCursor);
	
	paramsCursor += m_encoder->m_paramSize;
	gradCursor += m_encoder->m_paramSize;

	m_decoder->bindWeights(paramsCursor, gradCursor);
}

float RNNTranslator::computeGrad (float *grad, float *params, float *data, float *target, int minibatchSize) {
	int maxThreads = omp_get_max_threads();
	float error = 0.f;
	
	memset(grad, 0x00, sizeof(float)*m_paramSize);
	bindWeights(params, grad);

	float *sampleData = data;
	float *sampleTarget = target;

	/*** feed forward and feed backward ***/
	for (int dataIdx=0; dataIdx<minibatchSize; ++dataIdx) {
		// TODO
		int encoderSeqLen = m_encoder->m_maxSeqLen;
		int decoderSeqLen = m_decoder->m_maxSeqLen;

		/* reset internal states of LSTM layers */
		m_encoder->resetStates(encoderSeqLen);
		m_decoder->resetStates(decoderSeqLen);
		
		/****************************************************************
		*                      Feed Forward Phase                       *
		****************************************************************/
		// ********* encoder *********
		float *dataCursor = sampleData;		
		RecurrentLayer *enInputLayer  = m_encoder->m_vecLayers[0];
		RecurrentLayer *enOutputLayer = m_encoder->m_vecLayers[m_encoder->m_numLayer-1];
		// bind input sequence to m_inputActs of the input layer of the encoder
		if (m_reverseEncoder) {
			for (int seqIdx=encoderSeqLen; seqIdx>=1; --seqIdx) {
				memcpy(enInputLayer->m_inputActs[seqIdx], dataCursor, sizeof(float)*m_encoder->m_inputSize);
				dataCursor += m_encoder->m_inputSize;
			}
		} else {			
			for (int seqIdx=1; seqIdx<=encoderSeqLen; ++seqIdx) {
				memcpy(enInputLayer->m_inputActs[seqIdx], dataCursor, sizeof(float)*m_encoder->m_inputSize);
				dataCursor += m_encoder->m_inputSize;
			}
		}
		m_encoder->feedForward(encoderSeqLen);

		// ********* decoder *********
		// set the internal states of the decoder at t' = 0 to the internal states of encoder at t = T
		#pragma omp parallel for
		for (int layerIdx=1; layerIdx<m_encoder->m_numLayer; layerIdx++) {
			RNNLSTMLayer *enLayer = dynamic_cast<RNNLSTMLayer*>(m_encoder->m_vecLayers[layerIdx]);
			RNNLSTMLayer *deLayer = dynamic_cast<RNNLSTMLayer*>(m_decoder->m_vecLayers[layerIdx]);
			memcpy(deLayer->m_states[0], enLayer->m_states[encoderSeqLen], sizeof(float) * deLayer->m_numNeuron);
			memcpy(deLayer->m_outputActs[0], enLayer->m_outputActs[encoderSeqLen], sizeof(float) * deLayer->m_numNeuron);
		}
		// set the input for the decoder
		float *targetCursor = sampleTarget;
		RecurrentLayer *deInputLayer  = m_decoder->m_vecLayers[0];
		RecurrentLayer *deOutputLayer = m_decoder->m_vecLayers[m_decoder->m_numLayer-1];
		// for seqIdx = 1, the input will be empty, which is similar to the END-OF-SENTENCE input in Ilya's work
		// for seqIdx = 2 to decoderSeqLen, bind input sequence to m_inputActs of the input layer of the decoder
		// note at time step t, the input vector should be the target vector at time step t-1
		for (int seqIdx=2; seqIdx<=decoderSeqLen; ++seqIdx) {
			if (m_decoder->m_taskType == "classification") {
				for (int classIdx=0; classIdx<m_decoder->m_outputSize; classIdx++) {
					if ( abs(targetCursor[classIdx] - 1.f) < 1e-6 ) {
						memcpy(deInputLayer->m_inputActs[seqIdx], sampleData+m_decoder->m_inputSize*classIdx, sizeof(float)*m_decoder->m_inputSize);
					}
				}
			} else if (m_decoder->m_taskType == "regression") {
				memcpy(deInputLayer->m_inputActs[seqIdx], targetCursor, sizeof(float)*m_decoder->m_inputSize);
			}
			targetCursor += m_decoder->m_outputSize;
		}				
		// decoder feedforward based on internal states at t' = 0 and input sequence from t' = 1 to T'
		m_decoder->feedForward(decoderSeqLen);
		
		/****************************************************************
		*                      Compute Error Phase                      *
		****************************************************************/
		error += m_decoder->computeError(sampleTarget, decoderSeqLen);
		
		/****************************************************************
		*                      Feed Backword Phase                      *
		****************************************************************/
		// ********* decoder *********
		targetCursor = sampleTarget; // reset the target cursor to sample target
		// bind target sequence to m_outputErrs of the output layer of the decoder
		for (int seqIdx=1; seqIdx<=decoderSeqLen; ++seqIdx) {			
			memcpy(deOutputLayer->m_outputErrs[seqIdx], targetCursor, sizeof(float)*m_decoder->m_outputSize);			
			targetCursor += m_decoder->m_outputSize;
		}
		// decoder feed backward based on taget sequence from t' = 1 to T'
		m_decoder->feedBackward(decoderSeqLen);

		// ********* encoder *********
		// set the internal deltas of the encoder at t = T+1 to the internal deltas of encoder at t' = 1
		#pragma omp parallel for
		for (int layerIdx=1; layerIdx<m_encoder->m_numLayer; layerIdx++) {
			RNNLSTMLayer *enLayer = dynamic_cast<RNNLSTMLayer*>(m_encoder->m_vecLayers[layerIdx]);
			RNNLSTMLayer *deLayer = dynamic_cast<RNNLSTMLayer*>(m_decoder->m_vecLayers[layerIdx]);
			memcpy(enLayer->m_inGateDelta[encoderSeqLen+1], deLayer->m_inGateDelta[1], sizeof(float) * deLayer->m_numNeuron);
			memcpy(enLayer->m_forgetGateDelta[encoderSeqLen+1], deLayer->m_forgetGateDelta[1], sizeof(float) * deLayer->m_numNeuron);
			memcpy(enLayer->m_outGateDelta[encoderSeqLen+1], deLayer->m_outGateDelta[1], sizeof(float) * deLayer->m_numNeuron);
			memcpy(enLayer->m_preGateStateDelta[encoderSeqLen+1], deLayer->m_preGateStateDelta[1], sizeof(float) * deLayer->m_numNeuron);
			memcpy(enLayer->m_forgetGateActs[encoderSeqLen+1], deLayer->m_forgetGateActs[1], sizeof(float) * deLayer->m_numNeuron);
		}
		// encoder feed backward based on deltas at t = T+1
		m_encoder->feedBackward(encoderSeqLen);

		// move cursor to new position
		sampleData += encoderSeqLen * m_encoder->m_inputSize;
		sampleTarget += decoderSeqLen * m_decoder->m_outputSize;
	}

	// normalization by number of input sequences and clip gradients to [-1, 1]
	float normFactor = 1.f / (float) minibatchSize;	
	#pragma omp parallel for
	for (int dim=0; dim<m_paramSize; ++dim) {
		grad[dim] *= normFactor;
		if (grad[dim] < -1.f) {
			grad[dim] = -1.f;
		} else if (grad[dim] > 1.f) {
			grad[dim] = 1.f;
		}
	}
	error *= normFactor;

	return error;
}