using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace EasyNNFramework {
    [Serializable]
    public static class WeightHandler {

        //returns 0 when no weight found
        public static float getWeight(Neuron startNeuron, Neuron endNeuron, NEAT network) {
            if (network.connectionList.TryGetValue(startNeuron.name+endNeuron.name, out Connection value)) {
                return value.weight;
            }
            return 0f;
        }

        //updates weight when already added
        public static void addWeight(Neuron startNeuron, Neuron endNeuron, float weight, NEAT network) {
            bool isAvailable = network.connectionList.ContainsKey(startNeuron.name+endNeuron.name);

            if (isAvailable) {
                network.connectionList[startNeuron.name+endNeuron.name].weight = weight;
            } else {
                network.connectionList.Add(startNeuron.name+endNeuron.name, new Connection(weight, startNeuron, endNeuron));
                network.connectionList = network.connectionList.OrderBy(o => o.Value.toNeuron.layer)
                    .ToDictionary(x => x.Key, x => x.Value);
            }
        }

        public static bool removeWeight(Neuron startNeuron, Neuron endNeuron, NEAT network) {
            bool isAvailable = network.connectionList.ContainsKey(startNeuron.name + endNeuron.name);

            if (isAvailable) {
                network.connectionList.Remove(startNeuron.name+endNeuron.name);
                return true;
            } else {
                return false;
            }
        }

        public static void removeAllConnections(Neuron n, NEAT network) {
            foreach (KeyValuePair<string, Connection> connection in network.connectionList.ToList()) {
                if (connection.Value.fromNeuron.Equals(n)) {
                    WeightHandler.removeWeight(n, connection.Value.toNeuron, network);
                } else if (connection.Value.toNeuron.Equals(n)) {
                    WeightHandler.removeWeight(connection.Value.fromNeuron, n, network);
                }
            }
        }
    }
}
