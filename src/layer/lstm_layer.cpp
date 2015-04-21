#include <glog/logging.h>
#include <stdlib.h>
#include <immintrin.h>
#include "lstm_layer.h"

using namespace std;

#define DEBUG_LSTM_LAYER 0

LSTMLayer::LSTMLayer(int numNeuron, int maxSeqLen, int inputSize) : RecurrentLayer(numNeuron, maxSeqLen, inputSize) {	
	DLOG_IF(INFO, DEBUG_LSTM_LAYER) << "LSTMLayer constructor." << endl;

	// resize all vectors
	resize(m_maxSeqLen);

	// allocate memory
	for (int seqIdx=0; seqIdx<m_maxSeqLen+2; ++seqIdx) {
		allocateMem(seqIdx);
	}

	for (int idx=0; idx<4; ++idx) {
		float *neuronSizeBuf = new float[m_numNeuron];
		m_neuronSizeBuf.push_back(neuronSizeBuf);
		float *inputSizeBuf = new float[m_inputSize];
		m_inputSizeBuf.push_back(inputSizeBuf);
	}
	m_derivBuf = new float[m_numNeuron];

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
	DLOG_IF(INFO, DEBUG_LSTM_LAYER) << "LSTMLayer deconstructor." << endl;	

	for (int seqIdx=0; seqIdx<m_maxSeqLen+2; ++seqIdx) {
		releaseMem(seqIdx);
	}

	for (int idx=0; idx<4; ++idx) {
		if (m_neuronSizeBuf[idx] != NULL) {delete [] m_neuronSizeBuf[idx];}
		if (m_inputSizeBuf[idx] != NULL) {delete [] m_inputSizeBuf[idx];}
	}

	if (m_derivBuf != NULL) {delete [] m_derivBuf;}
}

void LSTMLayer::initParams(float *params) {
	float multiplier = 0.08; // follow sequence to sequence translation
	for (int i=0; i<m_nParamSize; i++) {
		params[i] = 0.0006; //multiplier * SYM_UNIFORM_RAND;
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

	// four delta
	m_preGateStateDelta.resize(newSeqLen+2);
	m_inGateDelta.resize(newSeqLen+2);
	m_forgetGateDelta.resize(newSeqLen+2);
	m_outGateDelta.resize(newSeqLen+2);
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

	// four deltas
	m_preGateStateDelta[seqIdx] = new float [m_numNeuron];
	m_inGateDelta[seqIdx] = new float [m_numNeuron];
	m_forgetGateDelta[seqIdx] = new float [m_numNeuron];
	m_outGateDelta[seqIdx] = new float [m_numNeuron];
}

void LSTMLayer::releaseMem(int seqIdx) {
	// three gate units
	if (m_inGateActs[seqIdx] != NULL) {delete [] m_inGateActs[seqIdx];}
	if (m_forgetGateActs[seqIdx] != NULL) {delete [] m_forgetGateActs[seqIdx];}
	if (m_outGateActs[seqIdx] != NULL) {delete [] m_outGateActs[seqIdx];}

	// states related
	if (m_preOutGateActs[seqIdx] != NULL) {delete [] m_preOutGateActs[seqIdx];}
	if (m_states[seqIdx] != NULL) {delete [] m_states[seqIdx];}
	if (m_preGateStates[seqIdx] != NULL) {delete [] m_preGateStates[seqIdx];}	

	// states errors
	if (m_cellStateErrs[seqIdx] != NULL) {delete [] m_cellStateErrs[seqIdx];}

	// four deltas
	if (m_preGateStateDelta[seqIdx] != NULL) {delete [] m_preGateStateDelta[seqIdx];}
	if (m_inGateDelta[seqIdx] != NULL) {delete [] m_inGateDelta[seqIdx];}
	if (m_forgetGateDelta[seqIdx] != NULL) {delete [] m_forgetGateDelta[seqIdx];}
	if (m_outGateDelta[seqIdx] != NULL) {delete [] m_outGateDelta[seqIdx];}
}

void LSTMLayer::reshape(int newSeqLen) {
	DLOG_IF(INFO, DEBUG_LSTM_LAYER) << "reshape LSTMLayer from" << m_maxSeqLen << " to " << newSeqLen << endl;
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
	DLOG_IF(INFO, DEBUG_LSTM_LAYER) << "LSTMLayer resetStates" << endl;

	#pragma omp parallel for
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

		// four deltas at Time t=T+1
		memset(m_outGateDelta[seqIdx], 0x00, sizeof(float) * m_numNeuron);
		memset(m_preGateStateDelta[seqIdx], 0x00, sizeof(float) * m_numNeuron);
		memset(m_forgetGateDelta[seqIdx], 0x00, sizeof(float) * m_numNeuron);
		memset(m_inGateDelta[seqIdx], 0x00, sizeof(float) * m_numNeuron);
	}

	// call parent class resetStates
	RecurrentLayer::resetStates(inputSeqLen);
}

void LSTMLayer::computeGatesActs(int seqIdx) {

	int maxThreads = omp_get_max_threads();
	int blockSize;
	if (maxThreads >= 4) {
	 	blockSize = (m_numNeuron + (maxThreads / 4) - 1) / (maxThreads / 4);
	} else {
		blockSize = m_numNeuron;
	}
	#pragma omp parallel for
	for (int threadIdx=0; threadIdx<max(maxThreads, 4); ++threadIdx) {
		int blockIdx = threadIdx / 4;
		int startIdx = blockIdx * blockSize;
		int actualSize = min(blockSize, m_numNeuron-startIdx);

		switch (threadIdx%4) {
			case 0: {
				// compute input gate activation
				dot(m_inGateActs[seqIdx]+startIdx, W_i_h+startIdx*m_numNeuron, actualSize, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron, 1);
				elem_mul(m_inGateActs[seqIdx]+startIdx, W_i_c+startIdx, m_states[seqIdx-1]+startIdx, actualSize);
				sigm(m_inGateActs[seqIdx]+startIdx, m_inGateActs[seqIdx]+startIdx, actualSize);
				break;
			}
			case 1: {
				// compute forget gate activation
				dot(m_forgetGateActs[seqIdx]+startIdx, W_f_h+startIdx*m_numNeuron, actualSize, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron, 1);
				elem_mul(m_forgetGateActs[seqIdx]+startIdx, W_f_c+startIdx, m_states[seqIdx-1]+startIdx, actualSize);
				sigm(m_forgetGateActs[seqIdx]+startIdx, m_forgetGateActs[seqIdx]+startIdx, actualSize);
				break;
			}
			case 2: {
				// compute pre-gate states
				dot(m_preGateStates[seqIdx]+startIdx, W_c_h+startIdx*m_numNeuron, actualSize, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron, 1);
				tanh(m_preGateStates[seqIdx]+startIdx, m_preGateStates[seqIdx]+startIdx, actualSize);
				break;
			}
			case 3: {
				// compute output gate activation
				dot(m_outGateActs[seqIdx]+startIdx, W_o_h+startIdx*m_numNeuron, actualSize, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron, 1);
				break;
			}
		}
	}
}

void LSTMLayer::feedForward(int inputSeqLen) {

	double startTime = CycleTimer::currentSeconds();
	// parafor each time step from T to 1
	#pragma omp parallel for
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		// compute input gate activation
		dot(m_inGateActs[seqIdx], W_i_x, m_numNeuron, m_inputSize, m_inputActs[seqIdx], m_inputSize, 1);

		// compute forget gate activation
		dot(m_forgetGateActs[seqIdx], W_f_x, m_numNeuron, m_inputSize, m_inputActs[seqIdx], m_inputSize, 1);

		// compute pre-gate states
		dot(m_preGateStates[seqIdx], W_c_x, m_numNeuron, m_inputSize, m_inputActs[seqIdx], m_inputSize, 1);

		// compute output gate activation
		dot(m_outGateActs[seqIdx], W_o_x, m_numNeuron, m_inputSize, m_inputActs[seqIdx], m_inputSize, 1);
	}
	double endTime = CycleTimer::currentSeconds();
	DLOG_EVERY_N(INFO, 1) << "[" << google::COUNTER << "]" << "LSTMLayer feedForward paralleled time: " << endTime - startTime << endl;

	startTime = CycleTimer::currentSeconds();
	// for each time step from 1 to T
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		// compute input gate activation
		// dot(m_inGateActs[seqIdx], W_i_x, m_numNeuron, m_inputSize, m_inputActs[seqIdx], m_inputSize, 1);
		// dot(m_inGateActs[seqIdx], W_i_h, m_numNeuron, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron, 1);
		// elem_mul(m_inGateActs[seqIdx], W_i_c, m_states[seqIdx-1], m_numNeuron);
		// sigm(m_inGateActs[seqIdx], m_inGateActs[seqIdx], m_numNeuron);
		
		// compute forget gate activation
		// dot(m_forgetGateActs[seqIdx], W_f_x, m_numNeuron, m_inputSize, m_inputActs[seqIdx], m_inputSize, 1);
		// dot(m_forgetGateActs[seqIdx], W_f_h, m_numNeuron, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron, 1);
		// elem_mul(m_forgetGateActs[seqIdx], W_f_c, m_states[seqIdx-1], m_numNeuron);
		// sigm(m_forgetGateActs[seqIdx], m_forgetGateActs[seqIdx], m_numNeuron);

		// compute pre-gate states
		// dot(m_preGateStates[seqIdx], W_c_x, m_numNeuron, m_inputSize, m_inputActs[seqIdx], m_inputSize, 1);
		// dot(m_preGateStates[seqIdx], W_c_h, m_numNeuron, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron, 1);
		// tanh(m_preGateStates[seqIdx], m_preGateStates[seqIdx], m_numNeuron);

		computeGatesActs(seqIdx);

		// compute cell states
		elem_mul(m_states[seqIdx], m_forgetGateActs[seqIdx], m_states[seqIdx-1], m_numNeuron);
		elem_mul(m_states[seqIdx], m_inGateActs[seqIdx], m_preGateStates[seqIdx], m_numNeuron);

		// compute output gate activation
		// dot(m_outGateActs[seqIdx], W_o_x, m_numNeuron, m_inputSize, m_inputActs[seqIdx], m_inputSize, 1);
		// dot(m_outGateActs[seqIdx], W_o_h, m_numNeuron, m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron, 1);
		elem_mul(m_outGateActs[seqIdx], W_o_c, m_states[seqIdx], m_numNeuron);
		sigm(m_outGateActs[seqIdx], m_outGateActs[seqIdx], m_numNeuron);

		// compute pre-output-gate activation
		tanh(m_preOutGateActs[seqIdx], m_states[seqIdx], m_numNeuron);

		// compute output activation
		elem_mul(m_outputActs[seqIdx], m_outGateActs[seqIdx], m_preOutGateActs[seqIdx], m_numNeuron);
	}
	endTime = CycleTimer::currentSeconds();
	DLOG_EVERY_N(INFO, 1) << "[" << google::COUNTER << "]" << "LSTMLayer feedForward sequential time: " << endTime - startTime << endl;	
}

void LSTMLayer::computeOutputErrs (int seqIdx) {
	#pragma omp parallel for
	for (int idx=0; idx<4; ++idx) {
		memset(m_neuronSizeBuf[idx], 0x00, sizeof(float) * m_numNeuron);
		switch (idx) {
			case 0: {
				trans_dot(m_neuronSizeBuf[idx], W_i_h, m_numNeuron, m_numNeuron, m_inGateDelta[seqIdx+1], m_numNeuron, 1);
				break;
			}
			case 1: {
				trans_dot(m_neuronSizeBuf[idx], W_f_h, m_numNeuron, m_numNeuron, m_forgetGateDelta[seqIdx+1], m_numNeuron, 1);
				break;
			}
			case 2: {
				trans_dot(m_neuronSizeBuf[idx], W_c_h, m_numNeuron, m_numNeuron, m_preGateStateDelta[seqIdx+1], m_numNeuron, 1);
				break;
			}
			case 3: {
				trans_dot(m_neuronSizeBuf[idx], W_o_h, m_numNeuron, m_numNeuron, m_outGateDelta[seqIdx+1], m_numNeuron, 1);
				break;
			}
		}
	}

	int blockNum = m_numNeuron / 4;
	#pragma omp parallel for
	for (int idx=0; idx<4; ++idx) {
		int start = idx * blockNum;
		int end = start + blockNum;
		for (int neuronIdx=start; neuronIdx<end; neuronIdx+=SIMD_WIDTH) {
			__m256 vec_0, vec_1, vec_2, vec_3, vec_res;
			vec_0 = _mm256_loadu_ps(m_neuronSizeBuf[0] + neuronIdx);
			vec_1 = _mm256_loadu_ps(m_neuronSizeBuf[1] + neuronIdx);
			vec_2 = _mm256_loadu_ps(m_neuronSizeBuf[2] + neuronIdx);
			vec_3 = _mm256_loadu_ps(m_neuronSizeBuf[3] + neuronIdx);
			vec_res = _mm256_loadu_ps(m_outputErrs[seqIdx] + neuronIdx);

			vec_res = _mm256_add_ps(vec_res, vec_0);
			vec_res = _mm256_add_ps(vec_res, vec_1);
			vec_res = _mm256_add_ps(vec_res, vec_2);
			vec_res = _mm256_add_ps(vec_res, vec_3);
			_mm256_storeu_ps(m_outputErrs[seqIdx] + neuronIdx, vec_res);
		}
	}

	// int residual = m_numNeuron % SIMD_WIDTH;
	// int stopSIMD = m_numNeuron - residual;

	// #pragma omp parallel for
	// for (int i=0; i<stopSIMD; i+=SIMD_WIDTH) {
	// 	__m256 vec_0, vec_1, vec_2, vec_3, vec_res;
	// 	vec_0 = _mm256_loadu_ps(m_neuronSizeBuf[0] + i);
	// 	vec_1 = _mm256_loadu_ps(m_neuronSizeBuf[1] + i);
	// 	vec_2 = _mm256_loadu_ps(m_neuronSizeBuf[2] + i);
	// 	vec_3 = _mm256_loadu_ps(m_neuronSizeBuf[3] + i);
	// 	vec_res = _mm256_loadu_ps(m_outputErrs[seqIdx] + i);

	// 	vec_res = _mm256_add_ps(vec_res, vec_0);
	// 	vec_res = _mm256_add_ps(vec_res, vec_1);
	// 	vec_res = _mm256_add_ps(vec_res, vec_2);
	// 	vec_res = _mm256_add_ps(vec_res, vec_3);
	// 	_mm256_storeu_ps(m_outputErrs[seqIdx] + i, vec_res);
	// }

	// for (int i=stopSIMD; i<m_numNeuron; ++i) {
	// 	m_outputErrs[seqIdx][i] += m_neuronSizeBuf[0][i];
	// 	m_outputErrs[seqIdx][i] += m_neuronSizeBuf[1][i];
	// 	m_outputErrs[seqIdx][i] += m_neuronSizeBuf[2][i];
	// 	m_outputErrs[seqIdx][i] += m_neuronSizeBuf[3][i];
	// }
}

void LSTMLayer::feedbackSequential (int seqIdx) {
	int blockSize = 16;
	#pragma omp parallel for
	for (int i=0; i<m_numNeuron; i+=blockSize) {
		int actualSize = min(blockSize, m_numNeuron-i);
		// computations are independent but use the same m_derivBuf
		// output gate delta (Time t = seqIdx): m_outGateDelta[seqIdx]
		sigm_deriv(m_derivBuf+i, m_outGateActs[seqIdx]+i, actualSize);
		elem_mul_triple(m_outGateDelta[seqIdx]+i, m_outputErrs[seqIdx]+i, m_derivBuf+i, m_preOutGateActs[seqIdx]+i, actualSize);

		// computations are independent but write to the same memory and depend on the seqIdx+1 time step
		// cell state error
		tanh_deriv(m_derivBuf+i, m_preOutGateActs[seqIdx]+i, actualSize);
		elem_mul_triple(m_cellStateErrs[seqIdx]+i, m_outputErrs[seqIdx]+i, m_outGateActs[seqIdx]+i, m_derivBuf+i, actualSize);

		elem_mul(m_cellStateErrs[seqIdx]+i, m_cellStateErrs[seqIdx+1]+i, m_forgetGateActs[seqIdx+1]+i, actualSize);
		elem_mul(m_cellStateErrs[seqIdx]+i, W_i_c+i, m_inGateDelta[seqIdx+1]+i, actualSize);
		elem_mul(m_cellStateErrs[seqIdx]+i, W_f_c+i, m_forgetGateDelta[seqIdx+1]+i, actualSize);
		elem_mul(m_cellStateErrs[seqIdx]+i, W_o_c+i, m_outGateDelta[seqIdx]+i, actualSize);

		// computations are independent but use the same m_derivBuf
		// pre-gate state delta (Time t = seqIdx): m_preGateStateDelta[seqIdx]
		tanh_deriv(m_derivBuf+i, m_preGateStates[seqIdx]+i, actualSize);
		elem_mul_triple(m_preGateStateDelta[seqIdx]+i, m_cellStateErrs[seqIdx]+i, m_inGateActs[seqIdx]+i, m_derivBuf+i, actualSize);

		// computations are independent but use the same m_derivBuf
		// forget gates delta (Time t = seqIdx): m_forgetGateDelta[seqIdx]
		sigm_deriv(m_derivBuf+i, m_forgetGateActs[seqIdx]+i, actualSize);
		elem_mul_triple(m_forgetGateDelta[seqIdx]+i, m_cellStateErrs[seqIdx]+i, m_states[seqIdx-1]+i, m_derivBuf+i, actualSize);

		// computations are independent but use the same m_derivBuf
		// input gates delta (Time t = seqIdx): m_inGateDelta[seqIdx]
		sigm_deriv(m_derivBuf+i, m_inGateActs[seqIdx]+i, actualSize);
		elem_mul_triple(m_inGateDelta[seqIdx]+i, m_cellStateErrs[seqIdx]+i, m_preGateStates[seqIdx]+i, m_derivBuf+i, actualSize);
	}
}

void LSTMLayer::feedBackward(int inputSeqLen) {

	double startTime = CycleTimer::currentSeconds();
	// sequential for each time step from T to 1
	for (int seqIdx=inputSeqLen; seqIdx>0; --seqIdx) {
		// four computations are independent but write to the same memory
		// output error: m_outputErrs[seqIdx]. all deltas are from Time t=seqIdx+1
		double seqBegTime = CycleTimer::currentSeconds();
		// computeOutputErrs (seqIdx);
		trans_dot(m_outputErrs[seqIdx], W_i_h, m_numNeuron, m_numNeuron, m_inGateDelta[seqIdx+1], m_numNeuron, 1);
		trans_dot(m_outputErrs[seqIdx], W_f_h, m_numNeuron, m_numNeuron, m_forgetGateDelta[seqIdx+1], m_numNeuron, 1);
		trans_dot(m_outputErrs[seqIdx], W_c_h, m_numNeuron, m_numNeuron, m_preGateStateDelta[seqIdx+1], m_numNeuron, 1);
		trans_dot(m_outputErrs[seqIdx], W_o_h, m_numNeuron, m_numNeuron, m_outGateDelta[seqIdx+1], m_numNeuron, 1);
		double seqEndTime = CycleTimer::currentSeconds();
		DLOG_EVERY_N(WARNING, 1) << "[" << google::COUNTER << "]" << "LSTMLayer feedBackward computeOutputErrs time: " << seqEndTime - seqBegTime << endl;

		seqBegTime = CycleTimer::currentSeconds();
		// feedbackSequential (seqIdx);		
		// computations are independent but use the same m_derivBuf
		// output gate delta (Time t = seqIdx): m_outGateDelta[seqIdx]
		sigm_deriv(m_derivBuf, m_outGateActs[seqIdx], m_numNeuron);
		elem_mul_triple(m_outGateDelta[seqIdx], m_outputErrs[seqIdx], m_derivBuf, m_preOutGateActs[seqIdx], m_numNeuron);
		printf("%d\n", seqIdx);
		for (int i=0; i<m_numNeuron; ++i) {
			printf("%f\t", m_outGateDelta[seqIdx][i]);
		}
		printf("\n");

		// computations are independent but write to the same memory and depend on the seqIdx+1 time step
		// cell state error
		tanh_deriv(m_derivBuf, m_preOutGateActs[seqIdx], m_numNeuron);
		elem_mul_triple(m_cellStateErrs[seqIdx], m_outputErrs[seqIdx], m_outGateActs[seqIdx], m_derivBuf, m_numNeuron);

		elem_mul(m_cellStateErrs[seqIdx], m_cellStateErrs[seqIdx+1], m_forgetGateActs[seqIdx+1], m_numNeuron);
		elem_mul(m_cellStateErrs[seqIdx], W_i_c, m_inGateDelta[seqIdx+1], m_numNeuron);
		elem_mul(m_cellStateErrs[seqIdx], W_f_c, m_forgetGateDelta[seqIdx+1], m_numNeuron);
		elem_mul(m_cellStateErrs[seqIdx], W_o_c, m_outGateDelta[seqIdx], m_numNeuron);

		// computations are independent but use the same m_derivBuf
		// pre-gate state delta (Time t = seqIdx): m_preGateStateDelta[seqIdx]
		tanh_deriv(m_derivBuf, m_preGateStates[seqIdx], m_numNeuron);
		elem_mul_triple(m_preGateStateDelta[seqIdx], m_cellStateErrs[seqIdx], m_inGateActs[seqIdx], m_derivBuf, m_numNeuron);

		// computations are independent but use the same m_derivBuf
		// forget gates delta (Time t = seqIdx): m_forgetGateDelta[seqIdx]
		sigm_deriv(m_derivBuf, m_forgetGateActs[seqIdx], m_numNeuron);
		elem_mul_triple(m_forgetGateDelta[seqIdx], m_cellStateErrs[seqIdx], m_states[seqIdx-1], m_derivBuf, m_numNeuron);

		// computations are independent but use the same m_derivBuf
		// input gates delta (Time t = seqIdx): m_inGateDelta[seqIdx]
		sigm_deriv(m_derivBuf, m_inGateActs[seqIdx], m_numNeuron);
		elem_mul_triple(m_inGateDelta[seqIdx], m_cellStateErrs[seqIdx], m_preGateStates[seqIdx], m_derivBuf, m_numNeuron);
		seqEndTime = CycleTimer::currentSeconds();
		DLOG_EVERY_N(WARNING, 1) << "[" << google::COUNTER << "]" << "LSTMLayer feedBackward feedbackSequential time: " << seqEndTime - seqBegTime << endl;
	}

	double endTime = CycleTimer::currentSeconds();
	DLOG_EVERY_N(INFO, 1) << "[" << google::COUNTER << "]" << "LSTMLayer feedBackward sequential time: " << endTime - startTime << endl;

	startTime = CycleTimer::currentSeconds();
	// omp parafor each time step from T to 1
	#pragma omp parallel for
	for (int seqIdx=inputSeqLen; seqIdx>0; --seqIdx) {
		// computations are independent but write to the same memory
		// spatial input error: m_inputErrs[seqIdx]
		trans_dot(m_inputErrs[seqIdx], W_i_x, m_numNeuron, m_inputSize, m_inGateDelta[seqIdx], m_numNeuron, 1);
		trans_dot(m_inputErrs[seqIdx], W_f_x, m_numNeuron, m_inputSize, m_forgetGateDelta[seqIdx], m_numNeuron, 1);
		trans_dot(m_inputErrs[seqIdx], W_c_x, m_numNeuron, m_inputSize, m_preGateStateDelta[seqIdx], m_numNeuron, 1);
		trans_dot(m_inputErrs[seqIdx], W_o_x, m_numNeuron, m_inputSize, m_outGateDelta[seqIdx], m_numNeuron, 1);

		// grad
		outer(gradW_i_x, m_inGateDelta[seqIdx], m_numNeuron, m_inputActs[seqIdx], m_inputSize);
		outer(gradW_i_h, m_inGateDelta[seqIdx], m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron);
		elem_mul(gradW_i_c, m_inGateDelta[seqIdx], m_states[seqIdx-1], m_numNeuron);

		outer(gradW_f_x, m_forgetGateDelta[seqIdx], m_numNeuron, m_inputActs[seqIdx], m_inputSize);
		outer(gradW_f_h, m_forgetGateDelta[seqIdx], m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron);
		elem_mul(gradW_f_c, m_forgetGateDelta[seqIdx], m_states[seqIdx-1], m_numNeuron);

		outer(gradW_c_x, m_preGateStateDelta[seqIdx], m_numNeuron, m_inputActs[seqIdx], m_inputSize);
		outer(gradW_c_h, m_preGateStateDelta[seqIdx], m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron);

		outer(gradW_o_x, m_outGateDelta[seqIdx], m_numNeuron, m_inputActs[seqIdx], m_inputSize);
		outer(gradW_o_h, m_outGateDelta[seqIdx], m_numNeuron, m_outputActs[seqIdx-1], m_numNeuron);
		elem_mul(gradW_o_c, m_outGateDelta[seqIdx], m_states[seqIdx-1], m_numNeuron);
	}

	endTime = CycleTimer::currentSeconds();
	DLOG_EVERY_N(INFO, 1) << "[" << google::COUNTER << "]" << "LSTMLayer feedBackward paralleled time: " << endTime - startTime << endl;	
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
