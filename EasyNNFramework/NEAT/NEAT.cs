using System;
using System.Collections.Generic;
using System.Linq;

namespace EasyNNFramework {
    [Serializable]
    public class NEAT {

        public LayerManager layerManager;
        public Dictionary<string, Connection> connectionList;

        public int counter = 0;

        //create a new neural network with predefined input and output neurons
        public NEAT(Dictionary<string, Neuron> _inputNeurons, Dictionary<string, Neuron> _actionNeurons) {
            layerManager = new LayerManager(_inputNeurons, _actionNeurons);
            connectionList = new Dictionary<string, Connection>();

            recalculateNeuronLayer();
        }

        //chances must add up to 100
        public void Mutate(System.Random rndObj, float chanceUpdateWeight, float chanceRemoveWeight, float chanceAddNeuron, float chanceRemoveNeuron, float chanceRandomFunction, ActivationFunction hiddenActivationFunction) {
            List<Neuron> possibleStartNeurons = new List<Neuron>();
            List<Neuron> possibleEndNeurons = new List<Neuron>();
            possibleStartNeurons.AddRange(layerManager.inputLayer.neurons.Values);
            possibleEndNeurons.AddRange(layerManager.actionLayer.neurons.Values);

            if (layerManager.layerCount > 2) {      //add hidden neurons if available
                possibleStartNeurons.AddRange(layerManager.getRandomHiddenLayer(rndObj).neurons.Values);
                possibleEndNeurons.AddRange(layerManager.getRandomHiddenLayer(rndObj).neurons.Values);
            }

            restart:
            //choose random start neuron
            Neuron rndStart = possibleStartNeurons[rndObj.Next(0, possibleStartNeurons.Count)];

            //rndObj end neuron
            Neuron rndEnd = possibleEndNeurons[rndObj.Next(0, possibleEndNeurons.Count)];

            //differentiate between neurons depending on layer
            //this is necessary when 2 hidden neurons were chosen as random neurons
            Neuron higherLayerNeuron, lowerLayerNeuron;
            if (rndStart.type == NeuronType.Input || rndEnd.type == NeuronType.Action) {
                lowerLayerNeuron = rndStart;
                higherLayerNeuron = rndEnd;
            } else if (rndStart.layer > rndEnd.layer) {
                higherLayerNeuron = rndStart;
                lowerLayerNeuron = rndEnd;
            } else if (rndStart.layer < rndEnd.layer) {
                higherLayerNeuron = rndEnd;
                lowerLayerNeuron = rndStart;
            } else {    //same layer
                goto restart;
            }

            float weight = WeightHandler.getWeight(lowerLayerNeuron, higherLayerNeuron, this);
            //if weight to rndObj neuron doesnt exist
            if (weight == 0f) {
                weight = UtilityClass.RandomWeight(rndObj);
            }

            bool recalcLayer = false, remUseless = false;
            //mutation options
            float rndChance = (float)rndObj.NextDouble();
            if (rndChance <= chanceUpdateWeight / 100f) {
                if (rndObj.NextDouble() <= 0.5f) {
                    //update weight with random value
                    WeightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, UtilityClass.RandomWeight(rndObj), this);
                } else {
                    //update weight with random multiplier
                    WeightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, UtilityClass.Clamp(-4f, 4f, weight * 2f * (float) rndObj.NextDouble()), this);
                }
            } else if (rndChance <= (chanceUpdateWeight + chanceRemoveWeight) / 100f) {
                WeightHandler.removeWeight(lowerLayerNeuron, higherLayerNeuron, this);
                remUseless = true;
                recalcLayer = true;
            } else if (rndChance <= (chanceUpdateWeight + chanceRemoveWeight + chanceAddNeuron) / 100f) {
                //add neuron between start and end
                Neuron newHidden = new Neuron(higherLayerNeuron.layer + "hidden" + counter, NeuronType.Hidden,
                    hiddenActivationFunction);

                Layer newLayer;
                if (higherLayerNeuron.layer - lowerLayerNeuron.layer > 1) {
                    newLayer = layerManager.getLayer(lowerLayerNeuron.layer + 1);
                } else {
                    newLayer = layerManager.addHiddenLayerBeforeAnother(layerManager.getLayer(higherLayerNeuron.layer));
                }
                newLayer.neurons.Add(newHidden.name, newHidden);
                counter++;
                WeightHandler.removeWeight(lowerLayerNeuron, higherLayerNeuron, this);
                WeightHandler.addWeight(lowerLayerNeuron, newHidden, weight, this);
                WeightHandler.addWeight(newHidden, higherLayerNeuron, 1, this);
                recalcLayer = true;
            } else if (rndChance <= (chanceUpdateWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron) / 100f) {
                if (layerManager.layerCount > 2) {
                    Layer rndLayer = layerManager.getRandomHiddenLayer(rndObj);
                    Neuron rndHidden = rndLayer.neurons.Values.ToList()[rndObj.Next(0, rndLayer.neurons.Count)];
                    WeightHandler.removeAllConnections(rndHidden, this);
                    rndLayer.neurons.Remove(rndHidden.name);
                    remUseless = true;
                    recalcLayer = true;
                }
            } else if (rndChance <= (chanceUpdateWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron + chanceRandomFunction) / 100f) {
                if (layerManager.layerCount > 2) {
                    Layer rndLayer = layerManager.getRandomHiddenLayer(rndObj);
                    Neuron rndHidden = rndLayer.neurons.Values.ToList()[rndObj.Next(0, rndLayer.neurons.Count)];
                    rndHidden.function = getRandomFunction(rndObj);
                }
            }

            if (remUseless && layerManager.layerCount > 2) {
                removeUselessHidden();
                removeUselessLayer();
            }

            if (recalcLayer) {
                recalculateNeuronLayer();
            }
        }

        public void recalculateNeuronLayer() {
            for (int i = 0; i < layerManager.allLayers.Count; i++) {
                foreach (Neuron n in layerManager.allLayers[i].neurons.Values) {
                    n.layer = i + 1;
                }
            }
        }

        private void removeUselessHidden() {

            string s = "";
            foreach (string key in connectionList.Keys) {
                s += key;
            }

            foreach (Layer hiddenLayer in layerManager.getAllHiddenLayers()) {
                foreach (Neuron hiddenNeuron in hiddenLayer.neurons.Values.ToList()) {
                    //if no connection found, remove
                    if (!s.Contains(hiddenNeuron.name)) {
                        hiddenLayer.neurons.Remove(hiddenNeuron.name);
                    }
                }
            }
        }

        private void removeUselessLayer() {
            foreach (Layer hiddenLayer in layerManager.getAllHiddenLayers().ToList()) {
                if (hiddenLayer.neurons.Count == 0) {
                    layerManager.allLayers.Remove(hiddenLayer);
                }
            }
        }

        public void calculateNetwork() {

            var vals = connectionList.Values.ToList();
            List<Connection> c = new List<Connection>();
            
            foreach (Layer layer in layerManager.allLayers) {
                if (layer.name == "input") {
                    continue;
                }

                foreach (Neuron neuron in layer.neurons.Values) {
                    //list containing all connections ending at current neuron
                    c.Clear();
                    for (int i = 0; i < vals.Count; i++) {
                        if (vals[i].toNeuron.Equals(neuron)) {
                            c.Add(vals[i]);
                        }
                    }

                    neuron.calculateValueWithIncommingConnections(c, this);
                }
            }
        }

        private ActivationFunction getRandomFunction(Random rnd) {
            return (ActivationFunction)rnd.Next(0, Enum.GetValues(typeof(ActivationFunction)).Length);
        }
    }

    public class Connection {

        public float weight;
        public Neuron fromNeuron, toNeuron;

        public Connection(float _weight, Neuron _fromNeuron, Neuron _toNeuron) {
            weight = _weight;
            fromNeuron = _fromNeuron;
            toNeuron = _toNeuron;
        }


    }
}
