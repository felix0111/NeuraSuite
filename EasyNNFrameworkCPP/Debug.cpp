#include "Debug.h"

#include <iostream>

#include "NEAT.h"

using namespace std;

int main(int argc, char* argv[])
{

	Neuron n = Neuron("1", INPUT, IDENTITY);
	Neuron n2 = Neuron("2", INPUT, IDENTITY);
	Neuron n3 = Neuron("3", INPUT, IDENTITY);
	Neuron n4 = Neuron("4", ACTION, SIGMOID);
	Neuron n5 = Neuron("5", ACTION, SIGMOID);
	Neuron n6 = Neuron("6", ACTION, IDENTITY);

	unordered_map<string, Neuron> in = { pair<string, Neuron>(n._name, n) , pair<string, Neuron>(n2._name, n2) , pair<string, Neuron>(n3._name, n3) };
	unordered_map<string, Neuron> out = { pair<string, Neuron>(n4._name, n4) , pair<string, Neuron>(n5._name, n5) , pair<string, Neuron>(n6._name, n6) };
	NEAT t = NEAT(in, out);

	t.AddConnection("1", "6", 2);
	t.AddHiddenNeuron("1", "6", 2, 1, IDENTITY);
	t.AddConnection("2", "hidden1", 1);

	t.GetNeuron("1")->_value = 20.0;
	t.GetNeuron("2")->_value = 20.0;

	t.UpdateLayerInformation();
	t.CalculateNetwork();

	cout << t.GetNeuron("6")->_value;
}
