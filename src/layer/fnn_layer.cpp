#include "fnn_layer.h"

using namespace std;

/****************************************************************
* Forward Layer
****************************************************************/

ForwardLayer::ForwardLayer(int numNeuron) {
	m_numNeuron = numNeuron;

	m_inputActs = new float[m_numNeuron];
	m_outputActs = new float[m_numNeuron];

	m_inputErrs = new float[m_numNeuron];
	m_outputErrs = new float[m_numNeuron];

	m_paramSize = 0;
}

ForwardLayer::~ForwardLayer() {
	if (m_inputActs != NULL) delete [] m_inputActs;
	if (m_outputActs != NULL) delete [] m_outputActs;
	if (m_inputErrs != NULL) delete [] m_inputErrs;
	if (m_outputErrs != NULL) delete [] m_outputErrs;
}

/****************************************************************
* Forward Softmax Layer
****************************************************************/

void FNNSoftmaxLayer::feedForward() {
	softmax(m_outputActs, m_inputActs, m_numNeuron);
}

void FNNSoftmaxLayer::feedBackward() {
	elem_sub(m_inputErrs, m_outputActs, m_outputErrs, m_numNeuron);
}