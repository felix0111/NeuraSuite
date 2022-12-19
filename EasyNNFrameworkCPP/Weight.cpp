#include "Weight.h"
#include <stdexcept>

Weight::Weight(std::string in, std::string out, double value)
{
	_in = in;
	_out = out;
	_value = value;
}