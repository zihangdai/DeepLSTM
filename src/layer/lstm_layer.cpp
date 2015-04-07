#include <stdlib.h>
#include "lstm_layer.h"

using namespace std;

LSTMLayer::LSTMLayer(int numNeuron, int maxSeqLen, int inputSize) : RecurrentLayer(numNeuron, maxSeqLen, inputSize) {
	#ifdef DEBUG_LSTM_LAYER
	printf("LSTMLayer constructor.\n");
	#endif

	// resize all vectors
	resize(m_maxSeqLen);

	// allocate memory
	for (int seqIdx=0; seqIdx<m_maxSeqLen+2; ++seqIdx) {
		allocateMem(seqIdx);
	}

	// four deltas
	m_preGateStateDelta = new float [m_numNeuron];
	m_inGateDelta = new float [m_numNeuron];
	m_forgetGateDelta = new float [m_numNeuron];
	m_outGateDelta = new float [m_numNeuron];

	// compute m_nParamSize
	m_nParamSize = 0;

	m_nParamSize += m_numNeuron * m_inputSize;	// W_i_x : [m_numNeuron x m_inputSize]
	m_nParamSize += m_numNeuron * m_numNeuron;	// W_i_h : [m_numNeuron x m_numNeuron]
	m_nParamSize += m_numNeuron;				// W_i_c : [m_numNeuron x 1]

	m_nParamSize += m_numNeuron * m_inputSize;	// W_f_x : [m_numNeuron x m_inputSize]
	m_nParamSize += m_numNeuron * m_numNeuron;	// W_f_c : [m_numNeuron x m_numNeuron]
	m_nParamSize += m_numNeuron;				// W_f_c : [m_numNeuron x 1]

	m_nParamSize += m_numNeuron * m_inputSize;	// W_c_x : [m_numNeuron x m_inputSize]
	m_nParamSize += m_numNeuron * m_numNeuron;	// W_c_h : [m_numNeuron x m_numNeuron]

	m_nParamSize += m_numNeuron * m_inputSize;	// W_o_x : [m_numNeuron x m_inputSize]
	m_nParamSize += m_numNeuron * m_numNeuron;	// W_o_h : [m_numNeuron x m_numNeuron]
	m_nParamSize += m_numNeuron;				// W_o_c : [m_numNeuron x 1]	
}

LSTMLayer::~LSTMLayer() {
	#ifdef DEBUG_LSTM_LAYER
	printf("LSTMLayer deconstructor.\n");
	#endif

	for (int seqIdx=0; seqIdx<m_maxSeqLen+2; ++seqIdx) {
		releaseMem(seqIdx);
	}
	// four deltas
	if (!m_preGateStateDelta) {delete [] m_preGateStateDelta;}
	if (!m_inGateDelta) {delete [] m_inGateDelta;}
	if (!m_forgetGateDelta) {delete [] m_forgetGateDelta;}
	if (!m_outGateDelta) {delete [] m_outGateDelta;}
}

void LSTMLayer::initParams(float *params) {
	float multiplier = 0.08; // follow sequence to sequence translation
	for (int i=0; i<m_nParamSize; i++) {
		params[i] = multiplier * SYM_UNIFORM_RAND;
	}
}

void LSTMLayer::resize(int newSeqLen) {
	// three gate units
	m_inGateActs.resize(newSeqLen+2);
	m_forgetGateActs.resize(newSeqLen+2);
	m_outGateActs.resize(newSeqLen+2);
	
	// states related
	m_preOutGateActs.resize(newSeqLen+2);
	m_states.resize(newSeqLen+2);
	m_preGateStates.resize(newSeqLen+2);
	
	// states errors
	m_cellStateErrs.resize(newSeqLen+2);
}

void LSTMLayer::allocateMem(int seqIdx) {
	// three gate units
	m_inGateActs[seqIdx] = new float [m_numNeuron];
	m_forgetGateActs[seqIdx] = new float [m_numNeuron];
	m_outGateActs[seqIdx] = new float [m_numNeuron];

	// states related
	m_preOutGateActs[seqIdx] = new float [m_numNeuron];
	m_states[seqIdx] = new float [m_numNeuron];
	m_preGateStates[seqIdx] = new float [m_numNeuron];

	// states errors
	m_cellStateErrs[seqIdx] = new float [m_numNeuron];
}

void LSTMLayer::releaseMem(int seqIdx) {
	// three gate units
	if (!m_inGateActs[seqIdx]) {delete [] m_inGateActs[seqIdx];}
	if (!m_forgetGateActs[seqIdx]) {delete [] m_forgetGateActs[seqIdx];}
	if (!m_outGateActs[seqIdx]) {delete [] m_outGateActs[seqIdx];}

	// states related
	if (!m_preOutGateActs[seqIdx]) {delete [] m_preOutGateActs[seqIdx];}
	if (!m_states[seqIdx]) {delete [] m_states[seqIdx];}
	if (!m_preGateStates[seqIdx]) {delete [] m_preGateStates[seqIdx];}	

	// states errors
	if (!m_cellStateErrs[seqIdx]) {delete [] m_cellStateErrs[seqIdx];}
}

void LSTMLayer::reshape(int newSeqLen) {
	#ifdef DEBUG_LSTM_LAYER
	printf("reshape LSTMLayer from %d to %d.\n", m_maxSeqLen, newSeqLen);
	#endif
	// release mem if needed
	if (newSeqLen < m_maxSeqLen) {
		for (int seqIdx=newSeqLen+2; seqIdx<m_maxSeqLen+2; ++seqIdx) {
			releaseMem(seqIdx);
		}
	}
	
	// resize vectors
	resize(newSeqLen);	

	// allocate new mem if needed
	if (newSeqLen > m_maxSeqLen) {
		for (int seqIdx=m_maxSeqLen+2; seqIdx<newSeqLen+2; ++seqIdx) {
			allocateMem(seqIdx);
		}
	}

	// call parent class reshape
	RecurrentLayer::reshape(newSeqLen);
}

void LSTMLayer::resetStates(int inputSeqLen) {
	/* all states and activations are initialised to zero at t = 0 */
	/* all delta and error terms are zero at t = T + 1 */
	#ifdef DEBUG_LSTM_LAYER
	printf("LSTMLayer resetStates.\n");
	#endif

	for (int seqIdx=0; seqIdx<inputSeqLen+2; ++seqIdx) {
		// three gate units
		memset(m_inGateActs[seqIdx], 0x00, sizeof(float) * m_numNeuron);
		memset(m_forgetGateActs[seqIdx], 0x00, sizeof(float) * m_numNeuron);
		memset(m_outGateActs[seqIdx], 0x00, sizeof(float) * m_numNeuron);

		// three states
		memset(m_preOutGateActs[seqIdx], 0x00, sizeof(float) * m_numNeuron);
		memset(m_states[seqIdx], 0x00, sizeof(float) * m_numNeuron);
		memset(m_preGateStates[seqIdx], 0x00, sizeof(float) * m_numNeuron);

		// cell errors
		memset(m_cellStateErrs[seqIdx], 0x00, sizeof(float) * m_numNeuron);	
	}

	// four deltas at Time t=T+1
	memset(m_outGateDelta, 0x00, sizeof(float) * m_numNeuron);
	memset(m_preGateStateDelta, 0x00, sizeof(float) * m_numNeuron);
	memset(m_forgetGateDelta, 0x00, sizeof(float) * m_numNeuron);
	memset(m_inGateDelta, 0x00, sizeof(float) * m_numNeuron);

	// call parent class resetStates
	RecurrentLayer::resetStates(inputSeqLen);
}

void LSTMLayer::feedForward(int inputSeqLen) {

	// for each time step from 1 to T
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		// compute input gate activation
		dot(m_inGateActs[seqIdx], W_i_x, m_numNeuron, m_inputSize, m_inputActs[seqIdx], m_inputSize, 1);
		dot(m_inGateActs[seqIdx], W_i_h, m_numNeuron, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron, 1);
		elem_mul(m_inGateActs[seqIdx], W_i_c, m_states[seqIdx-1], m_numNeuron);
		sigm(m_inGateActs[seqIdx], m_inGateActs[seqIdx], m_numNeuron);
		
		// compute forget gate activation
		dot(m_forgetGateActs[seqIdx], W_f_x, m_numNeuron, m_inputSize, m_inputActs[seqIdx], m_inputSize, 1);
		dot(m_forgetGateActs[seqIdx], W_f_h, m_numNeuron, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron, 1);
		elem_mul(m_forgetGateActs[seqIdx], W_f_c, m_states[seqIdx-1], m_numNeuron);
		sigm(m_forgetGateActs[seqIdx], m_forgetGateActs[seqIdx], m_numNeuron);

		// compute pre-gate states
		dot(m_preGateStates[seqIdx], W_c_x, m_numNeuron, m_inputSize, m_inputActs[seqIdx], m_inputSize, 1);
		dot(m_preGateStates[seqIdx], W_c_h, m_numNeuron, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron, 1);
		tanh(m_preGateStates[seqIdx], m_preGateStates[seqIdx], m_numNeuron);

		// compute cell states
		elem_mul(m_states[seqIdx], m_forgetGateActs[seqIdx], m_states[seqIdx-1], m_numNeuron);
		elem_mul(m_states[seqIdx], m_inGateActs[seqIdx], m_preGateStates[seqIdx], m_numNeuron);

		// compute output gate activation
		dot(m_outGateActs[seqIdx], W_o_x, m_numNeuron, m_inputSize, m_inputActs[seqIdx], m_inputSize, 1);
		dot(m_outGateActs[seqIdx], W_o_h, m_numNeuron, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron, 1);
		elem_mul(m_outGateActs[seqIdx], W_o_c, m_states[seqIdx], m_numNeuron);
		sigm(m_outGateActs[seqIdx], m_outGateActs[seqIdx], m_numNeuron);

		// compute pre-output-gate activation
		tanh(m_preOutGateActs[seqIdx], m_states[seqIdx], m_numNeuron);

		// compute output activation
		elem_mul(m_outputActs[seqIdx], m_outGateActs[seqIdx], m_preOutGateActs[seqIdx], m_numNeuron);
	}
}

void LSTMLayer::feedBackward(int inputSeqLen) {

	// allocate working buffer
	float *derivBuf = new float[m_numNeuron];

	// for each time step from T to 1
	for (int seqIdx=inputSeqLen; seqIdx>0; --seqIdx) {		
		
		// #pragma omp parallel
		// output error: m_outputErrs[seqIdx]. all deltas are from Time t=seqIdx+1
		trans_dot(m_outputErrs[seqIdx], W_i_h, m_numNeuron, m_numNeuron, m_inGateDelta, m_numNeuron, 1);
		trans_dot(m_outputErrs[seqIdx], W_f_h, m_numNeuron, m_numNeuron, m_forgetGateDelta, m_numNeuron, 1);
		trans_dot(m_outputErrs[seqIdx], W_c_h, m_numNeuron, m_numNeuron, m_preGateStateDelta, m_numNeuron, 1);
		trans_dot(m_outputErrs[seqIdx], W_o_h, m_numNeuron, m_numNeuron, m_outGateDelta, m_numNeuron, 1);

		// output gate delta (Time t = seqIdx): m_outGateDelta
		sigm_deriv(derivBuf, m_outGateActs[seqIdx], m_numNeuron);
		memset(m_outGateDelta, 0x00, sizeof(float) * m_numNeuron);
		elem_mul_triple(m_outGateDelta, m_outputErrs[seqIdx], derivBuf, m_preOutGateActs[seqIdx], m_numNeuron);
		
		// cell state error
		tanh_deriv(derivBuf, m_preOutGateActs[seqIdx], m_numNeuron);
		elem_mul_triple(m_cellStateErrs[seqIdx], m_outputErrs[seqIdx], m_outGateActs[seqIdx], derivBuf, m_numNeuron);
		elem_mul(m_cellStateErrs[seqIdx], m_cellStateErrs[seqIdx+1], m_forgetGateActs[seqIdx+1], m_numNeuron);
		elem_mul(m_cellStateErrs[seqIdx], W_i_c, m_inGateDelta, m_numNeuron);
		elem_mul(m_cellStateErrs[seqIdx], W_f_c, m_forgetGateDelta, m_numNeuron);
		elem_mul(m_cellStateErrs[seqIdx], W_o_c, m_outGateDelta, m_numNeuron);

		// pre-gate state delta (Time t = seqIdx): m_preGateStateDelta
		tanh_deriv(derivBuf, m_preGateStates[seqIdx], m_numNeuron);
		memset(m_preGateStateDelta, 0x00, sizeof(float) * m_numNeuron);
		elem_mul_triple(m_preGateStateDelta, m_cellStateErrs[seqIdx], m_inGateActs[seqIdx], derivBuf, m_numNeuron);

		// forget gates delta (Time t = seqIdx): m_forgetGateDelta
		sigm_deriv(derivBuf, m_forgetGateActs[seqIdx], m_numNeuron);
		memset(m_forgetGateDelta, 0x00, sizeof(float) * m_numNeuron);
		elem_mul_triple(m_forgetGateDelta, m_cellStateErrs[seqIdx], m_states[seqIdx-1], derivBuf, m_numNeuron);

		// input gates delta (Time t = seqIdx): m_inGateDelta
		sigm_deriv(derivBuf, m_inGateActs[seqIdx], m_numNeuron);
		memset(m_inGateDelta, 0x00, sizeof(float) * m_numNeuron);
		elem_mul_triple(m_inGateDelta, m_cellStateErrs[seqIdx], m_preGateStates[seqIdx], derivBuf, m_numNeuron);

		// spatial input error: m_inputErrs[seqIdx]
		trans_dot(m_inputErrs[seqIdx], W_i_x, m_numNeuron, m_inputSize, m_inGateDelta, m_numNeuron, 1);
		trans_dot(m_inputErrs[seqIdx], W_f_x, m_numNeuron, m_inputSize, m_forgetGateDelta, m_numNeuron, 1);
		trans_dot(m_inputErrs[seqIdx], W_c_x, m_numNeuron, m_inputSize, m_preGateStateDelta, m_numNeuron, 1);
		trans_dot(m_inputErrs[seqIdx], W_o_x, m_numNeuron, m_inputSize, m_outGateDelta, m_numNeuron, 1);

		// grad
		outer(gradW_i_x, m_inGateDelta, m_numNeuron, m_inputActs[seqIdx], m_inputSize);
		outer(gradW_i_h, m_inGateDelta, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron);
		elem_mul(gradW_i_c, m_inGateDelta, m_states[seqIdx-1], m_numNeuron);

		outer(gradW_f_x, m_forgetGateDelta, m_numNeuron, m_inputActs[seqIdx], m_inputSize);
		outer(gradW_f_h, m_forgetGateDelta, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron);
		elem_mul(gradW_f_c, m_forgetGateDelta, m_states[seqIdx-1], m_numNeuron);

		outer(gradW_c_x, m_preGateStateDelta, m_numNeuron, m_inputActs[seqIdx], m_inputSize);
		outer(gradW_c_h, m_preGateStateDelta, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron);

		outer(gradW_o_x, m_outGateDelta, m_numNeuron, m_inputActs[seqIdx], m_inputSize);
		outer(gradW_o_h, m_outGateDelta, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron);
		elem_mul(gradW_o_c, m_outGateDelta, m_states[seqIdx-1], m_numNeuron);

		// DEBUG CODE
		// for (int j=0; j<m_numNeuron; ++j) {
		// 	printf("m_outGateDelta(%d,%d):%f\n", seqIdx, j, m_outGateDelta[j]);
			// printf("m_preGateStateDelta(%d,%d):%f\n", seqIdx, j, m_preGateStateDelta[j]);
			// printf("m_forgetGateDelta(%d,%d):%f\n", seqIdx, j, m_forgetGateDelta[j]);
			// printf("m_inGateDelta(%d,%d):%f\n", seqIdx, j, m_inGateDelta[j]);
		// }
	}

	// delete working buffer
	delete [] derivBuf;
}

void LSTMLayer::bindWeights(float *params, float *grad) {
	// weights
	float *paramCursor = params;
	
	W_i_x = paramCursor; 						// [m_numNeuron x m_inputSize]
	paramCursor += m_numNeuron * m_inputSize;
	W_i_h = paramCursor; 						// [m_numNeuron x m_numNeuron]
	paramCursor += m_numNeuron * m_numNeuron;
	W_i_c = paramCursor; 						// [m_numNeuron x 1]
	paramCursor += m_numNeuron;

	W_f_x = paramCursor; 						// [m_numNeuron x m_inputSize]
	paramCursor += m_numNeuron * m_inputSize;
	W_f_h = paramCursor; 						// [m_numNeuron x m_numNeuron]
	paramCursor += m_numNeuron * m_numNeuron;
	W_f_c = paramCursor; 						// [m_numNeuron x 1]
	paramCursor += m_numNeuron;

	W_c_x = paramCursor; 						// [m_numNeuron x m_inputSize]
	paramCursor += m_numNeuron * m_inputSize;
	W_c_h = paramCursor; 						// [m_numNeuron x m_numNeuron]
	paramCursor += m_numNeuron * m_numNeuron;						

	W_o_x = paramCursor; 						// [m_numNeuron x m_inputSize]
	paramCursor += m_numNeuron * m_inputSize;
	W_o_h = paramCursor; 						// [m_numNeuron x m_numNeuron]
	paramCursor += m_numNeuron * m_numNeuron;
	W_o_c = paramCursor; 						// [m_numNeuron x 1]

	// grad
	float *gradCursor = grad;
	gradW_i_x = gradCursor; 					// [m_numNeuron x m_inputSize]
	gradCursor += m_numNeuron * m_inputSize;
	gradW_i_h = gradCursor; 					// [m_numNeuron x m_numNeuron]
	gradCursor += m_numNeuron * m_numNeuron;
	gradW_i_c = gradCursor; 					// [m_numNeuron x 1]
	gradCursor += m_numNeuron;

	gradW_f_x = gradCursor; 					// [m_numNeuron x m_inputSize]
	gradCursor += m_numNeuron * m_inputSize;
	gradW_f_h = gradCursor; 					// [m_numNeuron x m_numNeuron]
	gradCursor += m_numNeuron * m_numNeuron;
	gradW_f_c = gradCursor; 					// [m_numNeuron x 1]
	gradCursor += m_numNeuron;

	gradW_c_x = gradCursor; 					// [m_numNeuron x m_inputSize]
	gradCursor += m_numNeuron * m_inputSize;
	gradW_c_h = gradCursor; 					// [m_numNeuron x m_numNeuron]
	gradCursor += m_numNeuron * m_numNeuron;						

	gradW_o_x = gradCursor; 					// [m_numNeuron x m_inputSize]
	gradCursor += m_numNeuron * m_inputSize;
	gradW_o_h = gradCursor; 					// [m_numNeuron x m_numNeuron]
	gradCursor += m_numNeuron * m_numNeuron;
	gradW_o_c = gradCursor; 					// [m_numNeuron x 1]
}
