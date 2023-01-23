using System;
using System.Collections.Generic;
using System.Runtime.Remoting.Messaging;

namespace EasyNNFramework.NEAT {

    [Serializable]
    public struct Neuron : IEquatable<Neuron> {
        public float value, lastValue;
        public ActivationFunction function;
        public readonly NeuronType type;
        public List<int> incommingConnections, outgoingConnections;
        public int ID, activationCount;

        public bool IsReady() {
            return activationCount == incommingConnections.Count;
        }

        public Neuron(int ID, ActivationFunction _function, NeuronType _type) {
            this.ID = ID;
            function = _function;
            type = _type;
            value = 0f;
            lastValue = 0f;
            incommingConnections = new List<int>();
            outgoingConnections = new List<int>();
            activationCount = 0;
        }

        private const float l = 1.0507009873554804934193349852946f;
        private const float a = 1.6732632423543772848170429916717f;
        public void processValue() {
            float sum = value;

            switch (function) {
                case ActivationFunction.GELU:
                    value = 0.5f * sum * (1 + (float)Math.Tanh(Math.Sqrt(2f / Math.PI) * (sum + 0.044715f * Math.Pow(sum, 3))));
                    break;
                case ActivationFunction.TANH:
                    value = (float)Math.Tanh(sum);
                    break;
                case ActivationFunction.SIGMOID:
                    value = 1.0f / (1.0f + (float)Math.Exp(-sum));
                    break;
                case ActivationFunction.SWISH:
                    value = sum / (1.0f + (float)Math.Exp(-sum));
                    break;
                case ActivationFunction.RELU:
                    value = Math.Max(0, sum);
                    break;
                case ActivationFunction.SELU:
                    value = sum > 0 ? l * sum : l * a * ((float)Math.Exp(sum) - 1f);
                    break;
                case ActivationFunction.IDENTITY:
                    value = sum;
                    break;
                default:
                    value = sum;
                    break;
            }
        }

        public override bool Equals(object obj) => obj is Neuron n && Equals(n);

        public static bool operator ==(Neuron lf, Neuron ri) => lf.Equals(ri);

        public static bool operator !=(Neuron lf, Neuron ri) => !(lf==ri);

        public override int GetHashCode() => ID.GetHashCode();

        public bool Equals(Neuron obj) => obj.ID == ID;
    }

    public enum NeuronType {Input, Hidden, Action}
    public enum ActivationFunction { GELU = 0, TANH = 1, SIGMOID = 2, SWISH = 3, RELU = 4, SELU = 5, IDENTITY = 6 }

}
