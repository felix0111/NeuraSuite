using System;
using System.Collections.Generic;

namespace EasyNNFramework.FeedForward {
    [Serializable]
    public class WeightHandler {
        
        public List<KeyValuePair<Neuron, Neuron>> neuronPairs;
        public List<float> weightList;

        public WeightHandler(NEAT _network) {
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
                startNeuron.outgoingConnections.Add(endNeuron.name);
                endNeuron.incommingConnections.Add(startNeuron.name);
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
                startNeuron.outgoingConnections.Remove(endNeuron.name);
                endNeuron.incommingConnections.Remove(endNeuron.name);
            } else {
                throw new KeyNotFoundException("Cannot find weight associated with " + startNeuron.name + " and " + endNeuron.name);
            }
        }
    }
}
