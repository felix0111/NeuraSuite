using System;
using System.Collections.Generic;
using System.Linq;

namespace EasyNNFramework {
    [Serializable]
    public class NEAT {

        public LayerManager layerManager;
        public Dictionary<string, Connection> connectionList;
        public Dictionary<string, Connection> recurrentConnectionList;

        public int counter = 0;

        //create a new neural network with predefined input and output neurons
        public NEAT(Dictionary<string, Neuron> _inputNeurons, Dictionary<string, Neuron> _actionNeurons) {
            layerManager = new LayerManager(_inputNeurons, _actionNeurons);
            connectionList = new Dictionary<string, Connection>();
            recurrentConnectionList = new Dictionary<string, Connection>();

            recalculateNeuronLayer();
        }

        //chances must add up to 100
        public void Mutate(System.Random rndObj, float chanceAddUpdateWeight, float chanceRandomizeWeight, float chanceRemoveWeight, float chanceAddNeuron, float chanceRemoveNeuron, float chanceRandomFunction, float chanceAddUpdateRecurrentWeight, ActivationFunction hiddenActivationFunction) {
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

            bool recalcLayer = false, remUseless = false;
            float rndChance = (float)rndObj.NextDouble();

            if (rndChance <= chanceAddUpdateWeight / 100f) {
                float weight = WeightHandler.getWeight(lowerLayerNeuron, higherLayerNeuron, this);
                float rndSign = rndObj.Next(0, 2) * 2 - 1;
                WeightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, UtilityClass.Clamp(-4f, 4f, weight + rndSign * (float)rndObj.NextDouble()), this);
            } else if (rndChance <= (chanceAddUpdateWeight + chanceRandomizeWeight) / 100f) {
                WeightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, UtilityClass.RandomWeight(rndObj), this);
            } else if (rndChance <= (chanceAddUpdateWeight + chanceRandomizeWeight + chanceRemoveWeight) / 100f) {
                if (recurrentConnectionList.Count != 0) {
                    if (rndObj.NextDouble() < 0.5D) {
                        Connection rndConnection = recurrentConnectionList.ElementAt(rndObj.Next(0, recurrentConnectionList.Count)).Value;
                        WeightHandler.removeWeight(rndConnection.fromNeuron, rndConnection.toNeuron, this);
                    } else {
                        Connection rndConnection = connectionList.ElementAt(rndObj.Next(0, connectionList.Count)).Value;
                        WeightHandler.removeWeight(rndConnection.fromNeuron, rndConnection.toNeuron, this);
                    }
                } else if (connectionList.Count != 0) {
                    Connection rndConnection = connectionList.ElementAt(rndObj.Next(0, connectionList.Count)).Value;
                    WeightHandler.removeWeight(rndConnection.fromNeuron, rndConnection.toNeuron, this);
                } else {
                    //do nothing because no connections
                }
                remUseless = true;
                recalcLayer = true;
            } else if (rndChance <= (chanceAddUpdateWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron) / 100f) {
                if (connectionList.Count != 0) {
                    Connection rndConnection = connectionList.ElementAt(rndObj.Next(0, connectionList.Count)).Value;
                    Layer newNeuronLayer;

                    if (rndConnection.toNeuron.layer - rndConnection.fromNeuron.layer > 1) {
                        newNeuronLayer = layerManager.getLayer(rndConnection.fromNeuron.layer + 1);
                    } else {
                        newNeuronLayer = layerManager.addHiddenLayerBeforeAnother(layerManager.getLayer(rndConnection.toNeuron.layer));
                    }

                    //create neuron between start and end
                    Neuron newHidden = new Neuron((rndConnection.fromNeuron.layer + 1) + "hidden" + counter, NeuronType.Hidden, hiddenActivationFunction);
                    newNeuronLayer.neurons.Add(newHidden.name, newHidden);
                    recalculateNeuronLayer();

                    counter++;

                    WeightHandler.addWeight(rndConnection.fromNeuron, newHidden, rndConnection.weight, this);
                    WeightHandler.addWeight(newHidden, rndConnection.toNeuron, 1, this);
                    WeightHandler.removeWeight(rndConnection.fromNeuron, rndConnection.toNeuron, this);

                    recalcLayer = true;
                }
            } else if (rndChance <= (chanceAddUpdateWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron) / 100f) {
                if (layerManager.layerCount > 2) {
                    Layer rndLayer = layerManager.getRandomHiddenLayer(rndObj);
                    Neuron rndHidden = rndLayer.neurons.ElementAt(rndObj.Next(0, rndLayer.neurons.Count)).Value;

                    WeightHandler.removeAllConnections(rndHidden, this);
                    rndLayer.neurons.Remove(rndHidden.name);
                    remUseless = true;
                    recalcLayer = true;
                }
            } else if (rndChance <= (chanceAddUpdateWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron + chanceRandomFunction) / 100f) {
                if (layerManager.layerCount > 2) {
                    Layer rndLayer = layerManager.getRandomHiddenLayer(rndObj);
                    Neuron rndHidden = rndLayer.neurons.ElementAt(rndObj.Next(0, rndLayer.neurons.Count)).Value;

                    rndHidden.function = getRandomFunction(rndObj);
                }
            } else if (rndChance <= (chanceAddUpdateWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron + chanceRandomFunction + chanceAddUpdateRecurrentWeight) / 100f) {
                if (layerManager.layerCount > 2) {
                    Neuron rndAction = layerManager.actionLayer.neurons.ElementAt(rndObj.Next(0, layerManager.actionLayer.neurons.Count)).Value;
                    Layer rndHiddenLayer = layerManager.getRandomHiddenLayer(rndObj);
                    Neuron rndHidden = rndHiddenLayer.neurons.ElementAt(rndObj.Next(0, rndHiddenLayer.neurons.Count)).Value;

                    WeightHandler.addWeight(rndAction, rndHidden, UtilityClass.RandomWeight(rndObj), this);
                }
            }

            if (remUseless && layerManager.layerCount > 2) {
                removeUselessHidden();
            }

            if (recalcLayer) {
                recalculateNeuronLayer();
            }
        }

        public void recalculateNeuronLayer() {
            for (int i = 1; i <= layerManager.allLayers.Count; i++) {
                foreach (Neuron n in layerManager.getLayer(i).neurons.Values) {
                    n.layer = i;
                }
            }
        }

        public void removeUselessHidden() {

            string startingS = "";
            string endingS = "";
            foreach (var c in connectionList.Values) {
                startingS += c.fromNeuron.name;
                endingS += c.toNeuron.name;
            }

            foreach (Layer hiddenLayer in layerManager.getAllHiddenLayers()) {
                foreach (Neuron hiddenNeuron in hiddenLayer.neurons.Values.ToList()) {
                    //if no connection with name of current neuron found, remove
                    if (!startingS.Contains(hiddenNeuron.name) || !endingS.Contains(hiddenNeuron.name)) {
                        WeightHandler.removeAllConnections(hiddenNeuron, this);
                        hiddenLayer.neurons.Remove(hiddenNeuron.name);
                    }
                }
            }

            removeUselessLayer();
        }

        private void removeUselessLayer() {
            foreach (Layer hiddenLayer in layerManager.getAllHiddenLayers().ToList()) {
                if (hiddenLayer.neurons.Count == 0) {
                    layerManager.allLayers.Remove(hiddenLayer);
                }
            }
            recalculateNeuronLayer();
        }

        public void calculateNetwork() {

            //reset all hidden
            for (int i = 2; i < layerManager.layerCount; i++) {
                foreach (Neuron neuron in layerManager.getLayer(i).neurons.Values) {
                    neuron.value = 0f;
                }
            }

            //apply recurrent
            foreach (Connection con in recurrentConnectionList.Values) {
                con.toNeuron.value += con.weight * con.fromNeuron.value;
            }

            //reset action layer
            foreach (Neuron neuron in layerManager.actionLayer.neurons.Values) {
                neuron.value = 0f;
            }

            //input layer is already always "calculated"
            int currentLayer = 2;
            do {

                //sum all connections going to current layer
                foreach (Connection connection in connectionList.Values) {
                    if (connection.toNeuron.layer == currentLayer) {
                        connection.toNeuron.value += connection.fromNeuron.value * connection.weight;
                    }
                }

                //run function for all neurons in current layer
                foreach (Neuron summedNeuron in layerManager.getLayer(currentLayer).neurons.Values) {
                    summedNeuron.value = Neuron.getFunctionValue(summedNeuron.function, summedNeuron.value);
                }

                //next layer
                currentLayer++;

            } while (currentLayer <= layerManager.layerCount);
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
