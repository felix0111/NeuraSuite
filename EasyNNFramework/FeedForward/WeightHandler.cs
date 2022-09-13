using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyNNFramework.FeedForward {
    internal class WeightHandler {

        private FFNN network;

        private List<KeyValuePair<Neuron, Neuron>> neuronPairs;
        private List<float> weightList;

        public WeightHandler(FFNN _network) {
            network = _network;

            neuronPairs = new List<KeyValuePair<Neuron, Neuron>>();
            weightList = new List<float>();
        }

        public float getWeight(Neuron startNeuron, Neuron endNeuron) {
            KeyValuePair<Neuron, Neuron> kvp = new KeyValuePair<Neuron, Neuron>(startNeuron, endNeuron);

            int index = neuronPairs.IndexOf(kvp);
            if (index != -1) {
                return weightList[index];
            }
            
            throw new KeyNotFoundException("Cannot find weight associated with " + startNeuron.name + " and " + endNeuron.name);
        }

        public void addWeight(Neuron startNeuron, Neuron endNeuron, float weight) {
            KeyValuePair<Neuron, Neuron> kvp = new KeyValuePair<Neuron, Neuron>(startNeuron, endNeuron);

            int index = neuronPairs.IndexOf(kvp);
            if (index == -1) {
                neuronPairs.Add(kvp);
                weightList.Add(weight);
            } else {
                weightList[index] = weight;
            }
        }

        public void removeWeight(Neuron startNeuron, Neuron endNeuron) {
            KeyValuePair<Neuron, Neuron> kvp = new KeyValuePair<Neuron, Neuron>(startNeuron, endNeuron);

            int index = neuronPairs.IndexOf(kvp);
            if (index != -1) {
                neuronPairs.RemoveAt(index);
                weightList.RemoveAt(index);
            } else {
                throw new KeyNotFoundException("Cannot find weight associated with " + startNeuron.name + " and " + endNeuron.name);
            }
        }
    }
}
