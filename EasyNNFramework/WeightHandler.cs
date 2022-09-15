using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace EasyNNFramework.FeedForward {
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

            bool exists = startNeuron.outgoingConnections.ContainsKey(endNeuron.name) && endNeuron.incommingConnections.ContainsKey(startNeuron.name);
            if (!exists) {
                startNeuron.outgoingConnections.Add(endNeuron.name, weight);
                endNeuron.incommingConnections.Add(startNeuron.name, weight);
            } else {
                startNeuron.outgoingConnections[endNeuron.name] = weight;
                endNeuron.incommingConnections[startNeuron.name] = weight;
            }
        }

        public static void removeWeight(Neuron startNeuron, Neuron endNeuron) {

            bool exists = startNeuron.outgoingConnections.ContainsKey(endNeuron.name) && endNeuron.incommingConnections.ContainsKey(startNeuron.name);
            if (exists) {
                startNeuron.outgoingConnections.Remove(endNeuron.name);
                endNeuron.incommingConnections.Remove(startNeuron.name);
            } else {
                throw new KeyNotFoundException("Cannot remove weight associated with " + startNeuron.name + " and " + endNeuron.name);
            }
        }
    }
}
