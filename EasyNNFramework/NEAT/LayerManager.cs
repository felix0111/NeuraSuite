using System;
using System.Collections.Generic;
using System.Linq;

namespace EasyNNFramework.NEAT {
    /*[Serializable]
    public struct LayerManager {

        public List<List<Neuron>> hiddenLayers;

        public LayerManager() {
            
        }

        public void addHiddenLayer(int layerIndex) {

            if (layerIndex < 0 || layerIndex >= hiddenLayers.Count) {
                throw new Exception("Layer index out of range!");
            }
            
            allLayers.Insert(layerIndex, new List<Neuron>());
        }

        public int getRandomHiddenLayer(System.Random rndObj) {
            if (layerCount < 3) throw new Exception("Couldn't get random hidden layer because there are no hidden layers!");
            return rndObj.Next(0, layerCount-2);
        }

        //layerIndexStart may be used to get even better performance by skipping layers
        public Neuron getNeuron(int ID, int layerIndexStart = 0) {
            for (int i = layerIndexStart; i < layerCount; i++) {
                for (int j = 0; j < allLayers[i].Count; j++) {
                    if (allLayers[i][j].ID == ID) return allLayers[i][j];
                }
            }

            throw new Exception("Could not find neuron: " + ID);
        }
    }*/
}
