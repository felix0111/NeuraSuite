#include "Neuron.h"

#include <stdexcept>

Neuron::Neuron()
{
	
}

Neuron::Neuron(std::string name, NeuronType type, ActivationFunction activationFunction)
{
	_name = name;
	_value = 0;
	_type = type;
    _activationFunction = activationFunction;
    layerIndex = 0;
    calculated = false;
}

double Neuron::GetFunctionValue(ActivationFunction function, double sum)
{
    switch (function) {
    case GELU:
        return 0.5 * sum * (1.0 + tanh(sqrt(2.0 / PI) * (sum + 0.044715f * pow(sum, 3))));
    case TANH:
        return tanh(sum);
    case SIGMOID:
        return 1.0 / (1.0 + exp(-sum));
    case SWISH:
        return sum / (1.0 + exp(-sum));
    case RELU:
        return std::max(0.0, sum);
    case SELU:
        if (sum > 0) return l * sum;
    	return l * a * (exp(sum) - 1.0);
    case IDENTITY:
        return sum;
    default:
        return sum;
    }
}

