using System;
using System.Collections.Generic;

namespace EasyNNFramework.NEAT {
    [Serializable]
    public class LayerManager {

        public List<Layer> allLayers;

        public List<Layer> hiddenLayers {
            get {
                if(allLayers.Count <= 2) return new List<Layer>();
                return allLayers.GetRange(1, allLayers.Count - 2);
            }
        }
        public Layer inputLayer => allLayers[0];
        public Layer actionLayer => allLayers[allLayers.Count - 1];
        public int layerCount => allLayers.Count;

        public LayerManager(Dictionary<int, Neuron> inputNeurons, Dictionary<int, Neuron> actionNeurons) {

            allLayers = new List<Layer> {
                new Layer() { neurons = inputNeurons },
                new Layer() { neurons = actionNeurons }
            };
        }

        public Layer addHiddenLayer(int layerIndex) {

            if (layerIndex < 1 || layerIndex >= layerCount) {
                throw new Exception("Layer index out of range!");
            }
            
            allLayers.Insert(layerIndex, new Layer());
            return allLayers[layerIndex];
        }

        public Layer getRandomHiddenLayer(System.Random rndObj) {
            if (layerCount < 3) throw new Exception("Couldn't get random hidden layer because there are no hidden layers!");
            return hiddenLayers[rndObj.Next(0, layerCount-2)];
        }

        //layerIndexStart may be used to get even better performance by skipping layers
        public Neuron getNeuron(int ID, int layerIndexStart = 0) {
            for (int i = layerIndexStart; i < layerCount; i++) {
                if (allLayers[i].neurons.TryGetValue(ID, out Neuron n)) return n;
            }

            throw new Exception("Could not find neuron: " + ID);
        }
    }

    [Serializable]
    public class Layer {
        public Dictionary<int, Neuron> neurons;

        public Layer() {
            neurons = new Dictionary<int, Neuron>();
        }
    }
}
