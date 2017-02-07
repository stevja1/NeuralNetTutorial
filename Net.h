#include <iostream>
#include <vector>
#include <cassert>
#include "Neuron.h"

/**
 * Net Class
 */
class Net
{
	public:
		Net(const vector<unsigned> &topology);
		void feedForward(const vector<double> &inputVals);
		void backProp(const vector<double> &targetVals);
		void getResults(vector<double> &resultVals) const;

	private:
		vector<vector<Neuron> > m_layers; // m_layers[layerNum][neuronNum]
		double m_error;
		double m_recentAverageError;
		double m_recentAverageSmoothingFactor;
};

void Net::getResults(vector<double> &resultVals) const
{
	resultVals.clear();

	for(unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::backProp(const vector<double> &targetVals)
{
	// Calculate overall net error (RMS of output neuron errors)
	vector<Neuron> &outputLayer = m_layers.back();
	m_error = 0.0;
	for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement:
	m_recentAverageError =
		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradients
	for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calculate grandients on hidden layers
	for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; -- layerNum)
	{
		vector<Neuron> &hiddenLayer = m_layers[layerNum];
		vector<Neuron> &nextLayer = m_layers[layerNum + 1];
		for(unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}
	// For all layers from outputs to first hidden layer,
	// update connection weights
	for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
		vector<Neuron> &layer = m_layers[layerNum];
		vector<Neuron> &prevLayer = m_layers[layerNum - 1];

		for(unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::feedForward(const vector<double> &inputVals)
{
	// Make sure that the number of inputs that we're receiving matches the number of
	// input neurons in our neural net (minus the bias neuron).
	assert(inputVals.size() == m_layers[0].size() - 1);
	// Assign (latch) the input values into the input neurons
	for(unsigned i = 0; i < inputVals.size(); ++i)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// Forward propigate -- start with 1 to skip the input layer
	for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
	{
		// Create a reference to the previous layer for the neuron's feed forward
		vector<Neuron> &prevLayer = m_layers[layerNum - 1];
		for(unsigned n = 0; n < m_layers[layerNum].size() -1; ++n)
		{
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

Net::Net(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(vector<Neuron>());
		unsigned numOutputs = layerNum == topology.size() -1 ? 0 : topology[layerNum +1];

		// We have made a new Layer, now fill it with neurons, and
		// add a bias neuron to the layer:
		for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "Made a Neuron!" << endl;
		}

		// Force the bias node's output value to 1.0. It's the last neuron created above;
		m_layers.back().back().setOutputVal(1.0);

	}
}
