using System;
using System.Net.Mail;

namespace EasyNNFramework.NEAT {

    [Serializable]
    public class Neuron : IEquatable<Neuron> {
        public float value;
        public ActivationFunction function;
        public int ID;
        public int LayerIndex(LayerManager lm) {
            for (int i = 0; i < lm.allLayers.Count; i++) {
                if (lm.allLayers[i].neurons.ContainsKey(ID)) return i;
            }

            throw new Exception("Could not find neuron ID in any layer!");
        }
        public NeuronType Type(LayerManager lm) {
            int index = LayerIndex(lm);
            if (index == 0) return NeuronType.Input;
            if (index == lm.allLayers.Count - 1) return NeuronType.Action;
            return NeuronType.Hidden;
        }

        public Neuron(int ID, ActivationFunction _function) {
            this.ID = ID;
            function = _function;
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

        public override bool Equals(object obj) {
            return this.Equals(obj as Neuron);
        }

        public static bool operator ==(Neuron lf, Neuron ri) {
            if (ReferenceEquals(lf, null)) return ReferenceEquals(ri, null);

            return lf.Equals(ri);
        }

        public static bool operator !=(Neuron lf, Neuron ri) => !(lf==ri);

        public override int GetHashCode() {
            return this.ID.GetHashCode();
        }

        public bool Equals(Neuron obj) {
            if (obj == null) return false;
            return obj.ID == ID;
        }
    }

    public enum NeuronType {Input, Hidden, Action}
    public enum ActivationFunction { GELU = 0, TANH = 1, SIGMOID = 2, SWISH = 3, RELU = 4, SELU = 5, IDENTITY = 6 }

}
