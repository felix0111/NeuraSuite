using System;
using System.Collections.Generic;
using System.Linq;
using EasyNNFramework.FeedForward;

namespace EasyNNFramework {

    [Serializable]
    public class Neuron : IEquatable<Neuron> {
        public readonly string name;
        public float value = 0f;
        public bool isCalculated = false;
        public Dictionary<string, float> incommingConnections, outgoingConnections;
        public NeuronType type;
        public ActivationFunction function;

        public Neuron(string _name, NeuronType _type, ActivationFunction _function) {
            name = _name;
            type = _type;
            function = _function;

            incommingConnections = new Dictionary<string, float>();
            outgoingConnections = new Dictionary<string, float>();
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

            if (isCalculated) {
                return value;
            }

            Neuron focused;
            float weight;
            foreach (KeyValuePair<string, float> incommingConnection in incommingConnections) {
                focused = network.getNeuronWithName(incommingConnection.Key);
                weight = incommingConnection.Value;

                if (focused.type == NeuronType.Input) {
                    sum += focused.value * weight;
                } else {
                    sum += focused.calculateValueWithIncomingConnections(network) * weight;
                }
            }

            value = getFunctionValue(function, sum);
            isCalculated = true;

            return value;
        }

        public int getLayer(NEAT network) {

            if (type == NeuronType.Action) {
                return network.highestLayer;
            }
            
            if (type == NeuronType.Input) {
                return 1;
            }

            int highestLayerInIncommingNeurons = 1;
            foreach (string key in incommingConnections.Keys) {
                int layer = network.getNeuronWithName(key).getLayer(network);
                if (layer > highestLayerInIncommingNeurons) {
                    highestLayerInIncommingNeurons = layer;
                }
            }
            
            return highestLayerInIncommingNeurons + 1;
        }

        public override bool Equals(object obj) {
            return this.Equals(obj as Neuron);
        }

        public override int GetHashCode() {
            return this.name.GetHashCode();
        }

        public bool Equals(Neuron obj) {
            if (obj == null) return false;
            return obj.name == name;
        }
    }

    public enum NeuronType {Input = 0, Hidden = 1, Action = 2}

    public enum ActivationFunction {GELU = 0, TANH = 1}
}
