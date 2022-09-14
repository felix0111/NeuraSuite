using System;
using System.Collections.Generic;
using EasyNNFramework.FeedForward;

namespace EasyNNFramework {

    [Serializable]
    public class Neuron {
        public string name;
        public float value = 0f;
        public List<string> incommingConnections, outgoingConnections;
        public NeuronType type;
        public ActivationFunction function;

        public Neuron(string _name, NeuronType _type, ActivationFunction _function) {
            name = _name;
            type = _type;
            function = _function;

            incommingConnections = new List<string>();
            outgoingConnections = new List<string>();
        }

        public static float getFunctionValue(ActivationFunction _function, float sum) {
            switch (_function) {
                case ActivationFunction.GELU:
                    return 0.5f * sum * (1 + (float)Math.Tanh(Math.Sqrt(2f / Math.PI) * (sum + 0.044715f * Math.Pow(sum, 3))));
                case ActivationFunction.TANH:
                    return (float)Math.Tanh(sum);
                default:
                    return sum;
            }
        }

        public float calculateValueWithIncomingConnections(NEAT network) {
            float sum = 0f;

            foreach (string incommingConnection in incommingConnections) {
                Neuron focused = network.getNeuronWithName(incommingConnection);

                if (focused.type == NeuronType.Input) {
                    sum += focused.value * network.weightHandler.getWeight(focused, this);
                } else {
                    sum += focused.calculateValueWithIncomingConnections(network) * network.weightHandler.getWeight(focused, this);
                }
            }

            sum = getFunctionValue(function, sum);
            value = sum;

            return sum;
        }

        public int getLayer(NEAT network) {

            if (type == NeuronType.Action) {
                return network.highestLayer;
            }

            if (type == NeuronType.Input) {
                return 1;
            }

            Neuron focused = network.getNeuronWithName(incommingConnections[0]);
            int thisLayer = focused.getLayer(network) + 1;

            if (network.highestLayer < thisLayer) {
                network.highestLayer = thisLayer;
            }
            return thisLayer;
        }
    }

    public enum NeuronType {Input = 0, Hidden = 1, Action = 2}

    public enum ActivationFunction {GELU = 0, TANH = 1}
}
