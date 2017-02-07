#include <cmath>
#include <cstdlib>
#include <vector>
#include "Connection.h"

using namespace std;

/**
 * Neuron Class
 */
class Neuron
{
	public:
		Neuron(unsigned numOutputs, unsigned myIndex);
		void setOutputVal(double val) { m_outputVal = val; }
		double getOutputVal(void) const { return m_outputVal; }
		void feedForward(const vector<Neuron> &prevLayer);
		void calcOutputGradients(double targetVal);
		void calcHiddenGradients(const vector<Neuron> &nextLayer);
		void updateInputWeights(vector<Neuron> &prevLayer);

	private:
		static double eta; // [0.0..1.0] overall net training rate
		static double alpha; // [0.0..1.0] multiplier of last weight change (momentum)
		static double transferFunction(double x);
		static double transferFunctionDerivative(double x);
		static double randomWeight(void) { return rand() / double(RAND_MAX); }
		double sumDOW(const vector<Neuron> &nextLayer) const;
		double m_outputVal;
		vector<Connection> m_outputWeights;
		unsigned m_myIndex;
		double m_gradient;
};

double Neuron::eta = 0.15; // overall net learing rate
double Neuron::alpha = 0.5; // momentum, multiplier of the last deltaWeight, [0.0..n]

void Neuron::updateInputWeights(vector<Neuron> &prevLayer)
{
	// The weights to be update are in the Connection container
	// in the neurons in the preceding layer

	for(unsigned n = 0; n < prevLayer.size(); ++n) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
			// Individual input, magnified by the gradient and train rate:
			eta
			* neuron.getOutputVal()
			* m_gradient
			// Also add momentum = a fraction of the previous delta weight
			+ alpha
			* oldDeltaWeight;

			neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
			neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const vector<Neuron> &nextLayer) const
{
	double sum;
	for(unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

void Neuron::calcHiddenGradients(const vector<Neuron> &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
	// tanh - output range [-1.0..1.0]
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	// tanh derivative estimation (only works for values of -1 < x < 1
	return 1.0 - x * x;
}

void Neuron::feedForward(const vector<Neuron> &prevLayer)
{
	double sum = 0.0;
	// Sum the previous layer's outputs (which are our inputs)
	// Include the bias node from the previous layer.

	for(unsigned n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() *
			prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for(unsigned c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Connection());
		// Make Connection a class that generates it's own number?
		// Let's be lazy and do one here.
		m_outputWeights.back().weight = randomWeight(); // Maybe call this method initializeWeight?
	}

	m_myIndex = myIndex;
}
