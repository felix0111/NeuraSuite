using System;
using System.Collections.Generic;

namespace EasyNNFramework {

    [Serializable]
    public class Neuron : IEquatable<Neuron> {
        public readonly string name;
        public float value = 0f;
        public int layer;
        public NeuronType type;
        public ActivationFunction function;

        public Neuron(string _name, NeuronType _type, ActivationFunction _function) {
            name = _name;
            type = _type;
            function = _function;
        }

        private const float l = 1.0507009873554804934193349852946f;
        private const float a = 1.6732632423543772848170429916717f;
        public static float getFunctionValue(ActivationFunction _function, float sum) {
            switch (_function) {
                case ActivationFunction.GELU:
                    return 0.5f * sum * (1 + (float)Math.Tanh(Math.Sqrt(2f / Math.PI) * (sum + 0.044715f * Math.Pow(sum, 3))));
                case ActivationFunction.TANH:
                    return (float)Math.Tanh(sum);
                case ActivationFunction.SIGMOID:
                    return 1.0f / (1.0f + (float)Math.Exp(-sum));
                case ActivationFunction.SWISH:
                    return sum / (1.0f + (float)Math.Exp(-sum));
                case ActivationFunction.RELU:
                    return Math.Max(0, sum);
                case ActivationFunction.SELU:
                    return sum > 0 ? l * sum : l * a * ((float) Math.Exp(sum) - 1f);
                case ActivationFunction.IDENTITY:
                    return sum;
                default:
                    return sum;
            }
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

    public enum ActivationFunction {GELU = 0, TANH = 1, SIGMOID = 2, SWISH = 3, RELU = 4, SELU = 5, IDENTITY = 6}

}
