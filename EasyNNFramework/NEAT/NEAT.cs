using System;
using System.Collections.Generic;
using System.Linq;

namespace EasyNNFramework {
    [Serializable]
    public class NEAT {

        public LayerManager layerManager;

        public int counter = 0;
        //public int highestLayer = 2;    //highest layer is normally 2 (input and output)

        //create a new neural network with predefined input and output neurons
        public NEAT(Dictionary<string, Neuron> _inputNeurons, Dictionary<string, Neuron> _actionNeurons) {
            layerManager = new LayerManager(_inputNeurons, _actionNeurons);
        }

        //chances must add up to 100
        public void Mutate(System.Random rndObj, float chanceUpdateWeight, float chanceRemoveWeight, float chanceAddNeuron, float chanceRemoveNeuron, float chanceRandomFunction) {
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
            } else if (rndStart.getLayerCount(this) > rndEnd.getLayerCount(this)) {
                higherLayerNeuron = rndStart;
                lowerLayerNeuron = rndEnd;
            } else if (rndStart.getLayerCount(this) < rndEnd.getLayerCount(this)) {
                higherLayerNeuron = rndEnd;
                lowerLayerNeuron = rndStart;
            } else {    //same layer
                goto restart;
            }

            float weight = WeightHandler.getWeight(lowerLayerNeuron, higherLayerNeuron);
            //if weight to rndObj neuron doesnt exist
            if (weight == 0f) {
                weight = UtilityClass.RandomWeight(rndObj);
                WeightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, weight);
            }

            //mutation options
            float rndChance = (float)rndObj.NextDouble();
            if (rndChance <= chanceUpdateWeight / 100f) {
                if (rndObj.NextDouble() <= 0.5f) {
                    //update weight with random value
                    WeightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, UtilityClass.RandomWeight(rndObj));
                } else {
                    //update weight with random multiplier
                    WeightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, UtilityClass.Clamp(-4f, 4f, weight * 2f * (float) rndObj.NextDouble()));
                }
            } else if (rndChance <= (chanceUpdateWeight + chanceRemoveWeight) / 100f) {
                WeightHandler.removeWeight(lowerLayerNeuron, higherLayerNeuron);
            } else if (rndChance <= (chanceUpdateWeight + chanceRemoveWeight + chanceAddNeuron) / 100f) {
                //add neuron between start and end
                Neuron newHidden = new Neuron(higherLayerNeuron.getLayerCount(this) + "hidden" + counter, NeuronType.Hidden,
                    ActivationFunction.GELU);

                Layer newLayer;
                if (higherLayerNeuron.getLayerCount(this) - lowerLayerNeuron.getLayerCount(this) > 1) {
                    newLayer = layerManager.getLayer(lowerLayerNeuron.getLayerCount(this) + 1);
                } else {
                    newLayer = layerManager.addHiddenLayerBeforeAnother(layerManager.getLayer(higherLayerNeuron.getLayerCount(this)));
                }
                newLayer.neurons.Add(newHidden.name, newHidden);
                counter++;
                WeightHandler.removeWeight(lowerLayerNeuron, higherLayerNeuron);
                WeightHandler.addWeight(lowerLayerNeuron, newHidden, weight);
                WeightHandler.addWeight(newHidden, higherLayerNeuron, 1);
            } else if (rndChance <= (chanceUpdateWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron) / 100f) {
                if (layerManager.layerCount > 2) {
                    Layer rndLayer = layerManager.getRandomHiddenLayer(rndObj);
                    Neuron rndHidden = rndLayer.neurons.Values.ToList()[rndObj.Next(0, rndLayer.neurons.Count)];
                    foreach (KeyValuePair<string, float> incomming in rndHidden.incommingConnections.ToList()) {
                        WeightHandler.removeWeight(getNeuronWithName(incomming.Key), rndHidden);
                    }
                    foreach (KeyValuePair<string, float> outgoing in rndHidden.outgoingConnections.ToList()) {
                        WeightHandler.removeWeight(rndHidden, getNeuronWithName(outgoing.Key));
                    }
                    rndLayer.neurons.Remove(rndHidden.name);
                }
            } else if (rndChance <= (chanceUpdateWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron + chanceRandomFunction) / 100f) {
                Layer rndLayer = layerManager.getRandomHiddenLayer(rndObj);
                Neuron rndHidden = rndLayer.neurons.Values.ToList()[rndObj.Next(0, rndLayer.neurons.Count)];
                rndHidden.function = getRandomFunction(rndObj);
            }

            if (layerManager.layerCount > 2) {
                removeUselessHidden();
                removeUselessLayer();
            }
        }

        private void removeUselessHidden() {
            foreach (Layer hiddenLayer in layerManager.getAllHiddenLayers()) {
                foreach (Neuron hiddenNeuron in hiddenLayer.neurons.Values.ToList()) {
                    if (hiddenNeuron.incommingConnections.Count == 0 || hiddenNeuron.outgoingConnections.Count == 0) {
                        foreach (string incoming in hiddenNeuron.incommingConnections.Keys.ToList()) {
                            WeightHandler.removeWeight(getNeuronWithName(incoming), hiddenNeuron);
                        }

                        foreach (string outgoing in hiddenNeuron.outgoingConnections.Keys.ToList()) {
                            WeightHandler.removeWeight(hiddenNeuron, getNeuronWithName(outgoing));
                        }

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
            
            foreach (Layer layer in layerManager.allLayers) {
                if (layer.name == "input") {
                    continue;
                }

                foreach (Neuron layerNeuron in layer.neurons.Values) {
                    layerNeuron.calculateValueWithIncomingConnections(this);
                }
            }
        }

        public Neuron getNeuronWithName(string name) {

            if (name.Contains("hidden")) {
                foreach (Layer hiddenLayer in layerManager.getAllHiddenLayers()) {
                    if (hiddenLayer.neurons.ContainsKey(name)) {
                        return hiddenLayer.neurons[name];
                    }
                }
            }

            if (layerManager.inputLayer.neurons.ContainsKey(name)) {
                return layerManager.inputLayer.neurons[name];
            }
            if (layerManager.actionLayer.neurons.ContainsKey(name)) {
                return layerManager.actionLayer.neurons[name];
            }

            throw new Exception("Can't find neuron with name!");
        }

        private ActivationFunction getRandomFunction(Random rnd) {
            return (ActivationFunction)rnd.Next(0, Enum.GetValues(typeof(ActivationFunction)).Length);
        }
    }
}
