#include "NEAT.h"

#include <ctime>

using namespace std;

NEAT::NEAT(unordered_map<string, Neuron> inputNeurons, unordered_map<string, Neuron> actionNeurons)
{
	_inputNeurons = inputNeurons;
	_actionNeurons = actionNeurons;
	counter = 0;
}

void NEAT::AddConnection(string from, string to, double value, bool changeIfExists)
{
	try
	{
		Weight *w = GetConnection(from, to);
		if(changeIfExists)
		{
			w->_value = value;
		}
	}
	catch (exception e)
	{
		_weights.emplace(to, Weight(from, to, value));
	}

	
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
	AddConnection(from, name, weightToHidden, false);
	AddConnection(name, to, weightOffHidden, false);
	counter++;
}

void NEAT::RemoveHiddenNeuron(Neuron *n)
{
	_hiddenLayers[n->layerIndex].erase(n->_name);
}

void NEAT::RemoveHiddenNeuron(std::string name)
{
	_hiddenLayers[GetNeuron(name)->layerIndex].erase(name);
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

void NEAT::Mutate(double chanceAddWeight, double chanceRandomizeWeight, double chanceUpdateWeight, double chanceRemoveWeight, double chanceAddNeuron, double chanceRemoveNeuron, double chanceRandomFunction, ActivationFunction hiddenActivationFunction, double rndNumber)
{
	srand(time(NULL) + rand());

	if(rndNumber <= chanceAddWeight)
	{
		restart:
		Neuron in;
		Neuron out;
		if(_hiddenLayers.empty())
		{
			in = *GetRandomNeuron(_inputNeurons);
			out = *GetRandomNeuron(_actionNeurons);
		} else
		{
			rand() % 2 == 0 ? in = *GetRandomNeuron(_inputNeurons) : in = *GetRandomNeuron(_hiddenLayers[rand() % _hiddenLayers.size()]);
			rand() % 2 == 0 ? out = *GetRandomNeuron(_hiddenLayers[rand() % _hiddenLayers.size()]) : out = *GetRandomNeuron(_actionNeurons);
		}

		if(in.layerIndex >= out.layerIndex)
		{
			goto restart;
		}

		AddConnection(in._name, out._name, (rand() % 400) / 100.0, false);

	} else if(rndNumber <= chanceAddWeight + chanceRandomizeWeight)
	{
		auto it = _weights.begin();
		advance(it, rand() % _weights.size());
		it->second._value = (rand() % 400) / 100.0;
	}
	else if (rndNumber <= chanceAddWeight + chanceRandomizeWeight + chanceUpdateWeight)
	{
		auto it = _weights.begin();
		advance(it, rand() % _weights.size());
		double multiplier = (800.0 + rand() % 400) / 100.0;
		it->second._value = max(-400.0, min(it->second._value*multiplier, 400.0));
	}
	else if (rndNumber <= chanceAddWeight + chanceRandomizeWeight + chanceUpdateWeight + chanceRemoveWeight)
	{
		auto it = _weights.begin();
		advance(it, rand() % _weights.size());
		RemoveConnection(it->second._in, it->second._out);
	}
	else if (rndNumber <= chanceAddWeight + chanceRandomizeWeight + chanceUpdateWeight + chanceRemoveWeight + chanceAddNeuron)
	{
		auto it = _weights.begin();
		advance(it, rand() % _weights.size());
		AddHiddenNeuron(it->second._in, it->second._out, it->second._value, 1.0, hiddenActivationFunction);
	}
	else if (rndNumber <= chanceAddWeight + chanceRandomizeWeight + chanceUpdateWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron)
	{
		RemoveHiddenNeuron(GetRandomNeuron(_hiddenLayers[rand() % _hiddenLayers.size()]));
	}
	else if (rndNumber <= chanceAddWeight + chanceRandomizeWeight + chanceUpdateWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron + chanceRandomFunction)
	{
		Neuron* rnd;
		rand() % 2 == 0 ? rnd = GetRandomNeuron(_hiddenLayers[rand() % _hiddenLayers.size()]) : rnd = GetRandomNeuron(_actionNeurons);
		rnd->_activationFunction = static_cast<ActivationFunction>(rand() % 7);
	}

	RemoveUseless();
	UpdateLayerInformation();
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

Neuron* NEAT::GetRandomNeuron(std::unordered_map<std::string, Neuron>& map)
{
	auto it = map.begin();
	advance(it, rand() % map.size());
	return &it->second;
}

void NEAT::RemoveUseless()
{
	unordered_multimap<string, string> commingFrom;
	unordered_multimap<string, string> goingTo;
	for (auto &weight : _weights)
	{
		commingFrom.emplace(weight.second._in, weight.second._in);
		goingTo.emplace(weight.second._out, weight.second._out);
	}

	//remove useless neurons
	for (int i = 0; i < _hiddenLayers.size(); i++)
	{
		//loop through all neurons
		auto it = _hiddenLayers[i].begin();
		while (it != _hiddenLayers[i].end())
		{
			//check if no connection on either side
			if(!commingFrom.count(it->first) || !goingTo.count(it->first))
			{
				it = _hiddenLayers[i].erase(it);
			} else
			{
				++it;
			}
		}
	}

	//remove useless layer
	auto it = _hiddenLayers.begin();
	while(it != _hiddenLayers.end())
	{
		if(it->empty())
		{
			it = _hiddenLayers.erase(it);
		} else
		{
			++it;
		}
	}
}
