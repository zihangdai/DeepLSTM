#include "rnn_layer.h"

using namespace std;

RecurrentLayer::RecurrentLayer (int numNeuron, int maxSeqLen, int inputSize) {
	m_numNeuron = numNeuron;
	m_maxSeqLen = maxSeqLen;
	m_inputSize = inputSize;

	// resize all vectors
	resize(m_maxSeqLen);

	// allocate memory for sequence of length T+2 (t=0 & t=T+1 are extra space for neat code)
	for (int seqIdx=0; seqIdx<m_maxSeqLen+2; ++seqIdx) {
		allocateMem(seqIdx);
	}

	m_paramSize = 0;
}

RecurrentLayer::~RecurrentLayer () {	
	// release memory for sequence of length T+2 (t=0 & t=T+1 are extra space for neat code)
	for (int seqIdx=0; seqIdx<m_maxSeqLen+2; ++seqIdx) {
		releaseMem(seqIdx);
	}
}

void RecurrentLayer::resetStates(int inputSeqLen) {
	#pragma omp parallel for
	for (int seqIdx=0; seqIdx<inputSeqLen+2; ++seqIdx) {
		memset(m_inputActs[seqIdx], 0x00, sizeof(float)*m_inputSize);
		memset(m_inputErrs[seqIdx], 0x00, sizeof(float)*m_inputSize);

		memset(m_outputActs[seqIdx], 0x00, sizeof(float)*m_numNeuron);
		memset(m_outputErrs[seqIdx], 0x00, sizeof(float)*m_numNeuron);
	}
}

void RecurrentLayer::resize (int newSeqLen) {
	m_inputActs.resize(newSeqLen+2);
	m_outputActs.resize(newSeqLen+2);

	m_inputErrs.resize(newSeqLen+2);
	m_outputErrs.resize(newSeqLen+2);
}

void RecurrentLayer::allocateMem (int seqIdx) {
	// m_inputActs and m_inputErrs
	m_inputActs[seqIdx] = new float [m_inputSize];
	m_inputErrs[seqIdx] = new float [m_inputSize];

	// m_outputActs and m_outputErrs
	m_outputActs[seqIdx] = new float [m_numNeuron];
	m_outputErrs[seqIdx] = new float [m_numNeuron];
}

void RecurrentLayer::releaseMem (int seqIdx) {
	// m_inputActs and m_inputErrs
	if (m_inputActs[seqIdx] != NULL) {delete [] m_inputActs[seqIdx];}
	if (m_inputErrs[seqIdx] != NULL) {delete [] m_inputErrs[seqIdx];}

	// m_outputActs and m_outputErrs
	if (m_outputActs[seqIdx] != NULL) {delete [] m_outputActs[seqIdx];}
	if (m_outputErrs[seqIdx] != NULL) {delete [] m_outputErrs[seqIdx];}
}

void RecurrentLayer::reshape(int newSeqLen) {
	// release mem if needed
	if (newSeqLen < m_maxSeqLen) {
		for (int seqIdx=newSeqLen+2; seqIdx<m_maxSeqLen+2; ++seqIdx) {
			releaseMem(seqIdx);
		}
	}	
	
	// resize all vectors
	resize(newSeqLen);	

	// allocate new mem if needed
	if (newSeqLen > m_maxSeqLen) {
		for (int seqIdx=m_maxSeqLen+2; seqIdx<newSeqLen+2; ++seqIdx) {
			allocateMem(seqIdx);
		}
	}

	// change parameter
	m_maxSeqLen = newSeqLen;
}