using System;
using System.Collections.Generic;
using System.Linq;

namespace EasyNNFramework.NEAT {

    [Serializable]
    public class Neuron : IEquatable<Neuron> {
        private float _sum;
        public float Value { get; private set; }
        public float LastValue { get; private set; }

        public ActivationFunction Function;
        public readonly NeuronType Type;
        public List<int> IncommingConnections, OutgoingConnections;

        public List<float> _inputs;

        public int ID { get; private set; }
        public bool Activated { get; private set; }

        public Neuron(int id, ActivationFunction function, NeuronType type) {
            ID = id;
            Function = function;
            Type = type;
            IncommingConnections = new List<int>();
            OutgoingConnections = new List<int>();
            _inputs = new List<float>();

            //defaults
            _sum = 0f;
            Value = 0f;
            LastValue = 0f;
            Activated = false;
        }

        private const float l = 1.0507009873554804934193349852946f;
        private const float a = 1.6732632423543772848170429916717f;
        public void Activate() {

            if (Function == ActivationFunction.MULT) {
                _sum = _inputs[0];
                for (int i = 1; i < _inputs.Count; i++) _sum *= _inputs[i];
            } else {
                _sum = _inputs.Sum();
            }

            switch (Function) {
                case ActivationFunction.GELU:
                    Value = 0.5f * _sum * (1 + (float)Math.Tanh(Math.Sqrt(2f / Math.PI) * (_sum + 0.044715f * Math.Pow(_sum, 3))));
                    break;
                case ActivationFunction.TANH:
                    Value = (float)Math.Tanh(_sum);
                    break;
                case ActivationFunction.SIGMOID:
                    Value = 1.0f / (1.0f + (float)Math.Exp(-_sum));
                    break;
                case ActivationFunction.SWISH:
                    Value = _sum / (1.0f + (float)Math.Exp(-_sum));
                    break;
                case ActivationFunction.RELU:
                    Value = Math.Max(0, _sum);
                    break;
                case ActivationFunction.SELU:
                    Value = _sum > 0 ? l * _sum : l * a * ((float)Math.Exp(_sum) - 1f);
                    break;
                case ActivationFunction.IDENTITY:
                    Value = _sum;
                    break;
                case ActivationFunction.LATCH:
                    Value = NNUtility.Latch(LastValue, _sum);
                    break;
                case ActivationFunction.ABS:
                    Value = Math.Abs(_sum);
                    break;
                case ActivationFunction.GAUSS:
                    Value = NNUtility.Gauss(_sum);
                    break;
                case ActivationFunction.MULT:
                    Value = _sum;
                    break;
                default:
                    Value = _sum;
                    break;
            }

            Activated = true;
        }

        public void Input(float value) {
            _inputs.Add(value);
        }

        public void ResetState() {
            _inputs.Clear();

            Activated = false;
            _sum = 0f;

            LastValue = Value;
            Value = 0f;
        }

        public override bool Equals(object obj) => Equals(obj as Neuron);

        public static bool operator ==(Neuron lf, Neuron ri) => lf.Equals(ri);

        public static bool operator !=(Neuron lf, Neuron ri) => !(lf==ri);

        public override int GetHashCode() => ID.GetHashCode();

        public bool Equals(Neuron obj) => obj != null && obj.ID == ID;

        //even though this is a struct, the two lists are ref type and need to be newly created
        public Neuron Clone() {
            Neuron clone = new Neuron(ID, Function, Type);
            clone.IncommingConnections = new List<int>(IncommingConnections);
            clone.OutgoingConnections = new List<int>(OutgoingConnections);
            return clone;
        }
    }

    public enum NeuronType {Input, Hidden, Action, Bias}
    public enum ActivationFunction { GELU = 0, TANH = 1, SIGMOID = 2, SWISH = 3, RELU = 4, SELU = 5, IDENTITY = 6, LATCH = 7, ABS = 8, GAUSS = 9, MULT = 10 }

}
