#ifndef __LAYER_H__
#define __LAYER_H__

#include <glog/logging.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

#include <vector>

#include "matrix.h"
#include "nonlinearity.h"
#include "cycle_timer.h"

using namespace std;

#define SYM_UNIFORM_RAND (2 * ((float) rand() / (RAND_MAX)) - 1)   // rand float in [-1, 1]

/****************************************************************
* Recurrent Layer
****************************************************************/
class RecurrentLayer
{
public:
	RecurrentLayer(int numNeuron, int maxSeqLen, int inputSize);
	virtual ~RecurrentLayer();

	/* data */
	int m_numNeuron;
	int m_inputSize;
	int m_maxSeqLen;

	int m_nParamSize;

	vector<float *> m_inputActs;
	vector<float *> m_outputActs;

	vector<float *> m_inputErrs;
	vector<float *> m_outputErrs;
	
	/* method */
	void virtual initParams(float *params) {};

	void virtual feedForward(int inputSeqLen) {};
	void virtual feedBackward(int inputSeqLen) {};

	void virtual forwardStep(int seqIdx) {};
	void virtual backwardStep(int seqIdx) {};

	void virtual bindWeights(float *params, float *grad) {};

	void virtual resetStates(int inputSeqLen);
	void virtual reshape(int newSeqLen);	

private:
	void resize (int newSeqLen);
	void releaseMem (int seqIdx);
	void allocateMem (int seqIdx);
};

/****************************************************************
* Forward Layer
****************************************************************/
class ForwardLayer
{
public:
	ForwardLayer() {};
	virtual ~ForwardLayer() {};

	/* data */
	int m_numNeuron;

	float *m_inputActs;
	float *m_outputActs;

	float *m_inputErrs;
	float *m_outputErrs;
	
	/* method */
	void virtual feedForward() {};
	void virtual feedBackward() {};
};

#endif