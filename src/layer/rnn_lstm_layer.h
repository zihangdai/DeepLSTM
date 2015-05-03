#ifndef __RNN_LSTM_LAYER_H__
#define __RNN_LSTM_LAYER_H__

#include "rnn_layer.h"

using namespace std;

class RNNLSTMLayer: public RecurrentLayer
{
public:
	RNNLSTMLayer(int numNeuron, int maxSeqLen, int inputSize);
	~RNNLSTMLayer();

	/* data */	

	// weight matrices
	float *W_i_x;
	float *W_i_h;
	float *W_i_c;
	float *W_f_x;
	float *W_f_h;
	float *W_f_c;
	float *W_c_x;
	float *W_c_h;
	float *W_o_x;
	float *W_o_h;
	float *W_o_c;

	float *Bias_i;
	float *Bias_f;
	float *Bias_c;
	float *Bias_o;

	// grad matrices
	float *gradW_i_x;
	float *gradW_i_h;
	float *gradW_i_c;
	float *gradW_f_x;
	float *gradW_f_h;
	float *gradW_f_c;
	float *gradW_c_x;
	float *gradW_c_h;
	float *gradW_o_x;
	float *gradW_o_h;
	float *gradW_o_c;

	float *gradBias_i;
	float *gradBias_f;
	float *gradBias_c;
	float *gradBias_o;

	// forward pass
	vector<float *> m_inGateActs;
	vector<float *> m_forgetGateActs;
	vector<float *> m_outGateActs;

	vector<float *> m_preOutGateActs;
	vector<float *> m_states;
	vector<float *> m_preGateStates;

	// backward pass
	vector<float *> m_cellStateErrs;
	
	vector<float *> m_preGateStateDelta;

	vector<float *> m_inGateDelta;
	vector<float *> m_forgetGateDelta;
	vector<float *> m_outGateDelta;

	vector<float *> m_neuronSizeBuf;
	vector<float *> m_inputSizeBuf;

	float *m_derivBuf;

	/* method */
	void initParams(float *params);

	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);
	void resetStates(int inputSeqLen);

	void reshape(int newSeqLen);
	void bindWeights(float *params, float *grad);

private:
	/* method */
	void resize (int newSeqLen);
	void releaseMem (int seqIdx);
	void allocateMem (int seqIdx);

	void computeOutputErrs(int seqIdx);
	void computeGatesActs(int seqIdx);
	void feedbackSequential (int seqIdx);
};

#endif