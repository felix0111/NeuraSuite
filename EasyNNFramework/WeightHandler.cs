using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyNNFramework.FeedForward {
    [Serializable]
    internal class WeightHandler {

        public List<KeyValuePair<Neuron, Neuron>> neuronPairs;
        private List<float> weightList;

        public WeightHandler() {
            neuronPairs = new List<KeyValuePair<Neuron, Neuron>>();
            weightList = new List<float>();
        }

        //returns 0 when no weight found
        public float getWeight(Neuron startNeuron, Neuron endNeuron) {
            KeyValuePair<Neuron, Neuron> kvp = new KeyValuePair<Neuron, Neuron>(startNeuron, endNeuron);

            int index = neuronPairs.IndexOf(kvp);
            if (index != -1) {
                return weightList[index];
            }

            return 0f;
        }

        public void addWeight(Neuron startNeuron, Neuron endNeuron, float weight) {
            KeyValuePair<Neuron, Neuron> kvp = new KeyValuePair<Neuron, Neuron>(startNeuron, endNeuron);

            int index = neuronPairs.IndexOf(kvp);
            if (index == -1) {
                neuronPairs.Add(kvp);
                weightList.Add(weight);
                startNeuron.outgoingConnections.Add(endNeuron);
                endNeuron.incommingConnections.Add(startNeuron);
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
                startNeuron.outgoingConnections.Remove(endNeuron);
                endNeuron.incommingConnections.Remove(endNeuron);
            } else {
                throw new KeyNotFoundException("Cannot find weight associated with " + startNeuron.name + " and " + endNeuron.name);
            }
        }
    }
}
