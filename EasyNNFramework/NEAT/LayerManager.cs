using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyNNFramework {
    [Serializable]
    public class LayerManager {
        
        public List<Layer> allLayers;

        public LayerManager(Dictionary<string, Neuron> _inputNeurons, Dictionary<string, Neuron> _actionNeurons) {

            allLayers = new List<Layer> {
                new Layer("input") { neurons = _inputNeurons },
                new Layer("action") { neurons = _actionNeurons }
            };
        }

        public Layer addHiddenLayerBeforeAnother(Layer another) {
            int indexAnother = allLayers.IndexOf(another);

            if (indexAnother <= 0) {
                throw new Exception("Can't add layer before input layer!");
            }

            Layer newHidden = new Layer("hidden");
            allLayers.Insert(indexAnother, newHidden);
            return newHidden;
        }

        public Layer getLayer(int layerCount) {
            return allLayers[layerCount-1];
        }

        public Layer getRandomHiddenLayer(System.Random rndObj) {
            return allLayers[rndObj.Next(1, allLayers.Count-1)];
        }

        public List<Layer> getAllHiddenLayers() {
            if (allLayers.Count > 2) {
                return allLayers.GetRange(1, allLayers.Count-2);
            }
            
            throw new Exception("No hidden layers found!");

        }

        public Layer getLayerFromNeuron(string name) {
            if (name.Contains("hidden")) {
                foreach (Layer hiddenLayer in getAllHiddenLayers()) {
                    if (hiddenLayer.neurons.ContainsKey(name)) {
                        return hiddenLayer;
                    }
                }
            }

            if (inputLayer.neurons.ContainsKey(name)) {
                return inputLayer;
            }
            if (actionLayer.neurons.ContainsKey(name)) {
                return actionLayer;
            }

            throw new Exception("Can't find neuron in any layer!");
        }

        public Layer inputLayer {
            get { return allLayers[0]; }
        }

        public Layer actionLayer {
            get { return allLayers[allLayers.Count-1]; }
        }

        public int layerCount {
            get { return allLayers.Count; }
        }
    }

    [Serializable]
    public class Layer {
        public Dictionary<string, Neuron> neurons;
        public readonly string name;

        public Layer (string _name) {
            name = _name;
            neurons = new Dictionary<string, Neuron>();
        }
    }
}
