using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using EasyNNFramework.FeedForward;

namespace EasyNNFramework {

    [Serializable]
    public class Neuron {
        public string name;
        public float value = 0f;
        public List<Neuron> incommingConnections;
        public List<Neuron> outgoingConnections;
        public NeuronType type;
        public ActivationFunction function;

        public Neuron(string _name, NeuronType _type, ActivationFunction _function) {
            name = _name;
            type = _type;
            function = _function;

            incommingConnections = new List<Neuron>();
            outgoingConnections = new List<Neuron>();
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

        public float calculateValueWithIncomingConnections(WeightHandler handler) {
            float sum = 0f;

            foreach (Neuron incommingConnection in incommingConnections) {
                if (incommingConnection.type == NeuronType.Input) {
                    sum += incommingConnection.value * handler.getWeight(incommingConnection, this);
                } else {
                    sum += incommingConnection.calculateValueWithIncomingConnections(handler) * handler.getWeight(incommingConnection, this);
                }
            }

            sum = getFunctionValue(function, sum);
            value = sum;

            return sum;
        }

        public int getLayer() {

            if (incommingConnections.Count == 0) {
                return 1;
            }
            return incommingConnections[0].getLayer() + 1;
        }
    }

    public enum NeuronType {Input = 0, Hidden = 1, Action = 2}

    public enum ActivationFunction {GELU = 0, TANH = 1}
}
