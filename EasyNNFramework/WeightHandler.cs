using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace EasyNNFramework {
    [Serializable]
    public static class WeightHandler {

        //returns 0 when no weight found
        public static float getWeight(Neuron startNeuron, Neuron endNeuron) {
            if (startNeuron.outgoingConnections.TryGetValue(endNeuron.name, out float value)) {
                return value;
            }
            return 0f;
        }

        //updates weight when already added
        public static void addWeight(Neuron startNeuron, Neuron endNeuron, float weight) {
            bool startNeuronHasEnd = startNeuron.outgoingConnections.ContainsKey(endNeuron.name);
            bool endNeuronHasStart = endNeuron.incommingConnections.ContainsKey(startNeuron.name);

            if (startNeuronHasEnd && endNeuronHasStart) {
                startNeuron.outgoingConnections[endNeuron.name] = weight;
                endNeuron.incommingConnections[startNeuron.name] = weight;
            } else if(startNeuronHasEnd != endNeuronHasStart) {
                throw new Exception("Corruption in weight system found when adding weight!");
            } else {
                startNeuron.outgoingConnections.Add(endNeuron.name, weight);
                endNeuron.incommingConnections.Add(startNeuron.name, weight);
            }
        }

        public static void removeWeight(Neuron startNeuron, Neuron endNeuron) {
            bool startNeuronHasEnd = startNeuron.outgoingConnections.ContainsKey(endNeuron.name);
            bool endNeuronHasStart = endNeuron.incommingConnections.ContainsKey(startNeuron.name);

            if (startNeuronHasEnd && endNeuronHasStart) {
                startNeuron.outgoingConnections.Remove(endNeuron.name);
                endNeuron.incommingConnections.Remove(startNeuron.name);
            } else if (startNeuronHasEnd != endNeuronHasStart){
                throw new KeyNotFoundException("Corruption in weight system found when removing weight!");
            } else {
                throw new Exception("Can't remove non-existing weight!");
            }
        }
    }
}
