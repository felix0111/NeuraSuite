#pragma once
#include <string>
#include <vector>

enum NeuronType
{
	INPUT, HIDDEN, ACTION
};

enum ActivationFunction
{
	GELU, TANH, SIGMOID, SWISH, RELU, SELU, IDENTITY
};

static const double PI = 3.141592653589793238463;
static const double l = 1.0507009873554804934193349852946;
static const double a = 1.6732632423543772848170429916717;

class Neuron
{
public:
	Neuron();
	Neuron(std::string name, NeuronType type, ActivationFunction activationFunction);

	static double GetFunctionValue(ActivationFunction function, double sum);

	std::string _name;
	double _value;
	NeuronType _type;
	ActivationFunction _activationFunction;
	int layerIndex;
	bool calculated;
};

