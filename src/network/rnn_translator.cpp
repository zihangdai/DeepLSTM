#include "rnn_translator.h"

RNNTranslator::RNNTranslator(ConfReader *confReader) {		
	// init encoder and decoder
	string encoderConfSec = confReader->getString("encoder_conf");
	string decoderConfSec = confReader->getString("decoder_conf");

	confReader *encoderConf = new confReader("config.conf", encoderConfSec);
	confReader *decoderConf = new confReader("config.conf", decoderConfSec);

	m_encoder = new lstm(encoderConf);	
	m_decoder = new lstm(decoderConf);
	
	// compute paramSize
	m_nParamSize = 0;
	m_nParamSize += m_encoder->m_nParamSize;
	m_nParamSize += m_decoder->m_nParamSize;
	m_nParamSize += m_encoder->m_targetSize * m_decoder->m_dataSize;

	// allocate memory for decode buffer
	m_codeSize = m_decoder->m_dataSize;
	m_decodeBuf = new float [m_codeSize * m_decoder->m_maxSeqLen];

}

RNNTranslator::~RNNTranslator() {
	if (!m_encoder) {
		delete m_encoder;
	}
	if (!m_decoder) {
		delete m_decoder;
	}
	if (!m_decodeBuf) {
		delete [] m_decodeBuf;
	}
}

void RNNTranslator::initParams (float *params) {
	float *paramsCursor = params;
	
	m_encoder->initParams(paramsCursor);
	paramsCursor += m_encoder->m_nParamSize;

	m_decoder->initParams(paramsCursor);
	paramsCursor += m_decoder->m_nParamSize;

	float multiplier = 0.08;
	for (int i=0; i<m_encoder->m_targetSize * m_decoder->m_dataSize; i++) {
		paramsCursor[i] = multiplier * SYM_UNIFORM_RAND;
	}
}

void RNNTranslator::bindWeights(float *params, float *grad) {
	float *paramsCursor = params;
	float *gradCursor = grad;
	
	m_encoder->bindWeights(paramsCursor, gradCursor);
	paramsCursor += m_encoder->m_nParamSize;
	gradCursor += m_encoder->m_nParamSize;

	m_decoder->bindWeights(paramsCursor, gradCursor);
	paramsCursor += m_decoder->m_nParamSize;
	gradCursor += m_decoder->m_nParamSize;

	m_encodingW = paramsCursor;
	m_gradEncodingW = gradCursor;
}

float RNNTranslator::computeGrad (float *grad, float *params, float *data, float *target, int minibatchSize) {
	float error = 0.f;
	
	memset(grad, 0x00, sizeof(float)*m_nParamSize);
	bindWeights(params, grad);

	float *sampleData = data;
	float *sampleTarget = target;

	/*** feed forward and feed backward ***/
	for (int dataIdx=0; dataIdx<minibatchSize; ++dataIdx) {
		// TODO
		int encoderSeqLen = m_encoder->m_maxSeqLen;
		int decoderSeqLen = m_decoder->m_maxSeqLen;		
		
		// feedforward
		// m_encoder->feedForward(sampleData, encoderSeqLen);
		// m_decoder->feedForward(, decoderSeqLen);
		// compute error
		

		// feedbackword
		// feedBackward(sampleTarget, inputSeqLen);

		/* reset internal states of LSTM layers */
		// resetStates(inputSeqLen); // this is subject to change

		// move cursor to new position
		sampleData += encoderSeqLen * m_encoder->m_dataSize;
		sampleTarget += decoderSeqLen * m_decoder->m_targetSize;
	}

	// normalization by number of input sequences and clip gradients to [-1, 1]
	float normFactor = 1.f / (float) minibatchSize;
	for (int dim=0; dim<m_nParamSize; ++dim) {
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