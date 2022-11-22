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

            string startingS = "";
            string endingS = "";
            foreach (var c in connectionList.Values) {
                startingS += c.fromNeuron.name;
                endingS += c.toNeuron.name;
            }

            foreach (Layer hiddenLayer in layerManager.getAllHiddenLayers()) {
                foreach (Neuron hiddenNeuron in hiddenLayer.neurons.Values.ToList()) {
                    //if no connection found, remove
                    if (!startingS.Contains(hiddenNeuron.name) || !endingS.Contains(hiddenNeuron.name)) {
                        WeightHandler.removeAllConnections(hiddenNeuron, this);
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

            //reset all hidden and action values
            for (int i = 2; i <= layerManager.layerCount; i++) {
                foreach (Neuron neuron in layerManager.getLayer(i).neurons.Values) {
                    neuron.value = 0f;
                }
            }

            //add sums and apply function
            List<Neuron> summed = new List<Neuron>();
            List<Connection> later = connectionList.Values.ToList();
            int currentLayer = 2;

            do {
                //sum all in current layer
                foreach (Connection connection in later.ToList()) {
                    if (connection.toNeuron.layer == currentLayer) {
                        connection.toNeuron.value += connection.fromNeuron.value * connection.weight;

                        if (!summed.Contains(connection.toNeuron)) summed.Add(connection.toNeuron);
                        later.Remove(connection);
                    }
                }

                //run function for all summed neurons in current layer
                foreach (Neuron summedNeuron in summed) {
                    summedNeuron.value = Neuron.getFunctionValue(summedNeuron.function, summedNeuron.value);
                }
                summed.Clear();

                //next layer
                currentLayer++;

            } while (later.Count != 0);
        }

        private ActivationFunction getRandomFunction(Random rnd) {
            return (ActivationFunction)rnd.Next(0, Enum.GetValues(typeof(ActivationFunction)).Length);
        }
    }

    [Serializable]
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
