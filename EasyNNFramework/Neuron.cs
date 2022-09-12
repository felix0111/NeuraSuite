using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyNNFramework {

    [Serializable]
    internal class Neuron {
        public string name;
        public float value = 0f;
        public NeuronType type;

        public Neuron(string _name, NeuronType _type) {
            name = _name;
            type = _type;
        }
    }

    public enum NeuronType {Input = 0, Hidden = 1, Action = 2}
}
