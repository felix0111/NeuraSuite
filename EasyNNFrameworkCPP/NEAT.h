#pragma once
#include <map>
#include <unordered_map>
#include <vector>
#include "Neuron.h"
#include "Weight.h"

class NEAT
{

public:
	//constructor
	NEAT(std::unordered_map<std::string, Neuron> inputNeurons, std::unordered_map<std::string, Neuron> actionNeurons);

	//misc functions
	void CalculateNetwork();
	void AddConnection(std::string from, std::string to, double value);
	void RemoveConnection(std::string from, std::string to);
	Weight *GetConnection(std::string from, std::string to);
	Neuron *GetNeuron(std::string name);
	void AddHiddenNeuron(std::string from, std::string to, double weightToHidden, double weightOffHidden, ActivationFunction activationFunction);
	void UpdateLayerInformation();

	std::unordered_map<std::string, Neuron> _inputNeurons;
	std::vector<std::unordered_map<std::string, Neuron>> _hiddenLayers;
	std::unordered_map<std::string, Neuron> _actionNeurons;
	std::multimap<std::string, Weight> _weights;

private:
	int counter;

	std::unordered_map<std::string, Neuron>* AddHiddenLayer(int atIndex);

};
