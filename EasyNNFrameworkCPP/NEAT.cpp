#include "NEAT.h"

using namespace std;

NEAT::NEAT(unordered_map<string, Neuron> inputNeurons, unordered_map<string, Neuron> actionNeurons)
{
	_inputNeurons = inputNeurons;
	_actionNeurons = actionNeurons;
	counter = 0;
}

void NEAT::AddConnection(string from, string to, double value)
{
	_weights.emplace(to, Weight(from, to, value));
}

void NEAT::RemoveConnection(string from, string to)
{
	auto its = _weights.equal_range(to);

	for (auto &it = its.first; it != its.second; it++)
	{
		if (it->second._in == from)
		{
			_weights.erase(it);
			break;
		}
	}
}

Weight *NEAT::GetConnection(string from, string to)
{
	auto its = _weights.equal_range(to);

	for (auto& it = its.first; it != its.second; it++)
	{
		if (it->second._in == from)
		{
			return &it->second;
		}
	}

	throw exception("Could not get reference to connection!");
}

Neuron* NEAT::GetNeuron(std::string name)
{
	if (_inputNeurons.count(name)) return &_inputNeurons[name];

	if (_actionNeurons.count(name)) return &_actionNeurons[name];

	for (auto it = _hiddenLayers.begin(); it != _hiddenLayers.end(); it++)
	{
		if (it->count(name)) return &(it->at(name));
	}

	throw exception("Could not find neuron by name!");
}

void NEAT::AddHiddenNeuron(string from, string to, double weightToHidden, double weightOffHidden, ActivationFunction activationFunction)
{
	string name = "hidden" + to_string(counter);

	pair<string, Neuron> pair = { name, Neuron(name, HIDDEN, activationFunction) };

	if(_hiddenLayers.empty())
	{
		pair.second.layerIndex = 0;
		AddHiddenLayer(0)->insert(pair);
	} else
	{
		Neuron* in = GetNeuron(from);
		Neuron* out = GetNeuron(to);

		//input to action
		if(in->_type == INPUT && out->_type == ACTION)
		{
			pair.second.layerIndex = out->layerIndex - 1;
			_hiddenLayers.back().insert(pair);
		}
		//input to hidden
		else if (in->_type == INPUT && out->_type == HIDDEN)
		{
			//input to first hidden layer
			if(out->layerIndex == 0)
			{
				//add new layer before first
				pair.second.layerIndex = 0;
				AddHiddenLayer(0)->insert(pair);
			}
			//input to other hidden layer
			else
			{
				pair.second.layerIndex = out->layerIndex - 1;
				_hiddenLayers[pair.second.layerIndex].insert(pair);
			}
		}
		//hidden to hidden
		else if(in->_type == HIDDEN && out->_type == HIDDEN)
		{
			//if no layer between
			if(in->layerIndex+1 == out->layerIndex)
			{
				pair.second.layerIndex = out->layerIndex;
				AddHiddenLayer(out->layerIndex)->insert(pair);
			}
			//if layer between
			else
			{
				pair.second.layerIndex = out->layerIndex - 1;
				_hiddenLayers[pair.second.layerIndex].insert(pair);
			}
		}
		//Hidden to action
		else if(in->_type == HIDDEN && out->_type == ACTION)
		{
			//if no layer between
			if (in->layerIndex + 1 == out->layerIndex)
			{
				pair.second.layerIndex = out->layerIndex;
				_hiddenLayers.emplace_back(unordered_map<string, Neuron>());
				_hiddenLayers.back().insert(pair);

			}
			//if layer between
			else
			{
				pair.second.layerIndex = out->layerIndex - 1;
				_hiddenLayers[pair.second.layerIndex].insert(pair);
			}
		}


	}

	RemoveConnection(from, to);
	AddConnection(from, name, weightToHidden);
	AddConnection(name, to, weightOffHidden);
	counter++;
}

std::unordered_map<std::string, Neuron>* NEAT::AddHiddenLayer(int atIndex)
{
	_hiddenLayers.insert(_hiddenLayers.begin()+atIndex, std::unordered_map<std::string, Neuron>());
	return &_hiddenLayers[atIndex];
}

void NEAT::UpdateLayerInformation()
{
	for (auto &neuron : _inputNeurons)
	{
		neuron.second.layerIndex = -1;
	}

	for (int i = 0; i < _hiddenLayers.size(); i++)
	{
		for (auto& neuron : _hiddenLayers[i])
		{
			neuron.second.layerIndex = i;
		}
	}

	for (auto& neuron : _actionNeurons)
	{
		neuron.second.layerIndex = _hiddenLayers.size();
	}
}

void NEAT::CalculateNetwork()
{
	if(_weights.empty())
	{
		for (pair<string, Neuron> pair : _actionNeurons)
		{
			pair.second._value = 0.0;
		}
		return;
	}

	for (auto &hiddenL : _hiddenLayers)
	{
		for (auto &hiddenNeuronPair : hiddenL)
		{
			if (hiddenNeuronPair.second.calculated) hiddenNeuronPair.second._value = 0.0; hiddenNeuronPair.second.calculated = false;

			auto allWeightsGoingToNeuron = _weights.equal_range(hiddenNeuronPair.first);
			for (auto &it = allWeightsGoingToNeuron.first; it != allWeightsGoingToNeuron.second; it++)
			{
				hiddenNeuronPair.second._value += it->second._value * GetNeuron(it->second._in)->_value;
			}
			hiddenNeuronPair.second._value = Neuron::GetFunctionValue(hiddenNeuronPair.second._activationFunction, hiddenNeuronPair.second._value);
			hiddenNeuronPair.second.calculated = true;
		}
	}

	for (auto &actionNeuronPair : _actionNeurons)
	{
		if (actionNeuronPair.second.calculated) actionNeuronPair.second._value = 0.0; actionNeuronPair.second.calculated = false;

		auto allWeightsGoingToNeuron = _weights.equal_range(actionNeuronPair.first);
		for (auto &it = allWeightsGoingToNeuron.first; it != allWeightsGoingToNeuron.second; it++)
		{
			actionNeuronPair.second._value += it->second._value * GetNeuron(it->second._in)->_value;
		}
		actionNeuronPair.second._value = Neuron::GetFunctionValue(actionNeuronPair.second._activationFunction, actionNeuronPair.second._value);
		actionNeuronPair.second.calculated = true;
	}
}