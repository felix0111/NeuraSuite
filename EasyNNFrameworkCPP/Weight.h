#pragma once
#include "Neuron.h"

class Weight
{

public:
	Weight(std::string in, std::string out, double value);
	double _value;
	std::string _in, _out;
};

