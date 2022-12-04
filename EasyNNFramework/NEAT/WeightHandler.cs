using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Messaging;

namespace EasyNNFramework {
    [Serializable]
    public static class WeightHandler {

        //returns 0 when no weight found
        public static float getWeight(Neuron startNeuron, Neuron endNeuron, NEAT network) {
            if (network.connectionList.TryGetValue(startNeuron.name + endNeuron.name, out Connection value)) {
                return value.weight;
            }
            return 0f;
        }

        //updates weight when already added
        public static void addWeight(Neuron startNeuron, Neuron endNeuron, float weight, NEAT network) {

            bool isAvailable;

            if (startNeuron.type == NeuronType.Action) {
                isAvailable = network.recurrentConnectionList.ContainsKey(startNeuron.name + endNeuron.name);
                if (isAvailable) {
                    network.recurrentConnectionList[startNeuron.name + endNeuron.name].weight = weight;
                } else {
                    network.recurrentConnectionList.Add(startNeuron.name + endNeuron.name, new Connection(weight, startNeuron, endNeuron));
                }

                return;
            } else if (startNeuron.layer >= endNeuron.layer) {
                throw new Exception("Can't add weight to lower layer neuron!");
            }

            isAvailable = network.connectionList.ContainsKey(startNeuron.name + endNeuron.name);
            if (isAvailable) {
                network.connectionList[startNeuron.name + endNeuron.name].weight = weight;
            } else {
                network.connectionList.Add(startNeuron.name + endNeuron.name, new Connection(weight, startNeuron, endNeuron));
                network.connectionList = network.connectionList.OrderBy(o => o.Value.toNeuron.layer)
                    .ToDictionary(x => x.Key, x => x.Value);
            }
        }

        public static bool removeWeight(Neuron startNeuron, Neuron endNeuron, NEAT network) {
            bool isAvailable;

            if (startNeuron.type == NeuronType.Action) {
                isAvailable = network.recurrentConnectionList.ContainsKey(startNeuron.name + endNeuron.name);
                if (isAvailable) {
                    network.recurrentConnectionList.Remove(startNeuron.name + endNeuron.name);
                    return true;
                } else {
                    return false;
                }
            } else if (startNeuron.layer >= endNeuron.layer) {
                throw new Exception("Can't remove weight because a connection to a lower layer neuron can never exist.");
            }


            isAvailable = network.connectionList.ContainsKey(startNeuron.name + endNeuron.name);
            if (isAvailable) {
                network.connectionList.Remove(startNeuron.name + endNeuron.name);
                return true;
            } else {
                return false;
            }
        }

        public static void removeAllConnections(Neuron n, NEAT network) {
            foreach (KeyValuePair<string, Connection> connection in network.connectionList.ToList()) {
                if (connection.Value.toNeuron.Equals(n) || connection.Value.fromNeuron.Equals(n)) {
                    network.connectionList.Remove(connection.Key);
                }
            }

            if (n.type == NeuronType.Input) return;

            //if not input neuron, then also check recurrent connections
            foreach (KeyValuePair<string, Connection> connection in network.recurrentConnectionList.ToList()) {
                if (connection.Value.fromNeuron.Equals(n) || connection.Value.toNeuron.Equals(n)) {
                    network.recurrentConnectionList.Remove(connection.Key);
                }
            }
        }
    }
}
