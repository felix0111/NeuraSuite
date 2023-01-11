using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace EasyNNFramework.NEAT {
    [Serializable]
    public class NEAT {

        public LayerManager layerManager;
        public Dictionary<int, List<Connection>> connectionList;
        public Dictionary<int, List<Connection>> recurrentConnectionList;

        public int IDCounter { get; private set; }

        public NEAT(Dictionary<int, Neuron> _inputNeurons, Dictionary<int, Neuron> _actionNeurons, int highestPredefinedNeuronID) {
            layerManager = new LayerManager(_inputNeurons, _actionNeurons);
            connectionList = new Dictionary<int, List<Connection>>();
            recurrentConnectionList = new Dictionary<int, List<Connection>>();
            IDCounter = highestPredefinedNeuronID + 1;
        }

        public NEAT(NEAT neat) {
            layerManager = new LayerManager(UtilityClass.CopyNeuronDictionary(neat.layerManager.inputLayer.neurons), UtilityClass.CopyNeuronDictionary(neat.layerManager.actionLayer.neurons));
            
            List<Layer> hiddensCopy = new List<Layer>(neat.layerManager.allLayers.Count - 2);
            foreach (Layer layer in neat.layerManager.hiddenLayers) {
                hiddensCopy.Add(new Layer() { neurons = UtilityClass.CopyNeuronDictionary(layer.neurons)} );
            }
            layerManager.allLayers.InsertRange(1, hiddensCopy);

            connectionList = UtilityClass.CopyConnectionDictionary(neat.connectionList);
            recurrentConnectionList = UtilityClass.CopyConnectionDictionary(neat.recurrentConnectionList);

            IDCounter = neat.IDCounter;
        }

        //chances must add up to 100
        public void Mutate(System.Random rndObj, float chanceAddWeight, float chanceRandomizeWeight, float chanceRemoveWeight, float chanceAddNeuron, float chanceRemoveNeuron, float chanceRandomFunction, float chanceAddRecurrentWeight, float chanceUpdateWeight, ActivationFunction hiddenActivationFunction) {
            List<Neuron> possibleStartNeurons = new List<Neuron>(layerManager.inputLayer.neurons.Values);
            List<Neuron> possibleEndNeurons = new List<Neuron>(layerManager.actionLayer.neurons.Values);

            //add hidden neurons if available
            if (layerManager.layerCount > 2) {
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
            int higherLayerNeuron, lowerLayerNeuron;
            if (rndStart.Type(layerManager) == NeuronType.Input || rndEnd.Type(layerManager) == NeuronType.Action) {
                lowerLayerNeuron = rndStart.ID;
                higherLayerNeuron = rndEnd.ID;
            } else if (rndStart.LayerIndex(layerManager) > rndEnd.LayerIndex(layerManager)) {
                higherLayerNeuron = rndStart.ID;
                lowerLayerNeuron = rndEnd.ID;
            } else if (rndStart.Type(layerManager) < rndEnd.Type(layerManager)) {
                higherLayerNeuron = rndEnd.ID;
                lowerLayerNeuron = rndStart.ID;
            } else {    //same layer
                goto restart;
            }

            bool remUseless = false;
            float rndChance = (float)rndObj.NextDouble();

            

            if (rndChance <= chanceAddWeight / 100f) {
                WeightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, UtilityClass.RandomWeight(rndObj), this);
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight) / 100f) {
                WeightHandler.updateWeight(lowerLayerNeuron, higherLayerNeuron, UtilityClass.RandomWeight(rndObj), this);
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight) / 100f) {
                if (recurrentConnectionList.Count != 0) {
                    if (rndObj.NextDouble() < 0.5D) {
                        KeyValuePair<int, List<Connection>> rndConnection = recurrentConnectionList.ElementAt(rndObj.Next(0, recurrentConnectionList.Count));
                        Connection rndTargetConnection = rndConnection.Value[rndObj.Next(0, rndConnection.Value.Count)];

                        WeightHandler.removeWeight(rndConnection.Key, rndTargetConnection.targetID, this);
                    } else {
                        KeyValuePair<int, List<Connection>> rndConnection = connectionList.ElementAt(rndObj.Next(0, connectionList.Count));
                        Connection rndTargetConnection = rndConnection.Value[rndObj.Next(0, rndConnection.Value.Count)];

                        WeightHandler.removeWeight(rndConnection.Key, rndTargetConnection.targetID, this);
                    }
                } else if (connectionList.Count != 0) {
                    KeyValuePair<int, List<Connection>> rndConnection = connectionList.ElementAt(rndObj.Next(0, connectionList.Count));
                    Connection rndTargetConnection = rndConnection.Value[rndObj.Next(0, rndConnection.Value.Count)];

                    WeightHandler.removeWeight(rndConnection.Key, rndTargetConnection.targetID, this);
                } else {
                    //do nothing because no connections
                }
                remUseless = true;
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron) / 100f) {
                if (connectionList.Count != 0) {
                    KeyValuePair<int, List<Connection>> rndConnection = connectionList.ElementAt(rndObj.Next(0, connectionList.Count));
                    Connection rndTargetConnection = rndConnection.Value[rndObj.Next(0, rndConnection.Value.Count)];

                    Neuron sourceNeuron = layerManager.getNeuron(rndConnection.Key);
                    Neuron targetNeuron = layerManager.getNeuron(rndTargetConnection.targetID);
                    Layer newNeuronLayer;

                    if (layerManager.getNeuron(rndTargetConnection.targetID).LayerIndex(layerManager) - layerManager.getNeuron(rndConnection.Key).LayerIndex(layerManager) > 1) {
                        newNeuronLayer = layerManager.allLayers[layerManager.getNeuron(rndConnection.Key).LayerIndex(layerManager) + 1];
                    } else {
                        newNeuronLayer = layerManager.addHiddenLayer(layerManager.getNeuron(rndConnection.Key).LayerIndex(layerManager) + 1);
                    }

                    //create neuron between start and end
                    Neuron newHidden = new Neuron(IDCounter, hiddenActivationFunction);
                    IDCounter++;

                    newNeuronLayer.neurons.Add(newHidden.ID, newHidden);

                    WeightHandler.addWeight(sourceNeuron.ID, newHidden.ID, rndTargetConnection.weight, this);
                    WeightHandler.addWeight(newHidden.ID, targetNeuron.ID, 1, this);
                    WeightHandler.removeWeight(sourceNeuron.ID, targetNeuron.ID, this);
                    remUseless = true;
                }
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron) / 100f) {
                if (layerManager.layerCount > 2) {
                    Layer rndLayer = layerManager.getRandomHiddenLayer(rndObj);
                    Neuron rndHidden = rndLayer.neurons.ElementAt(rndObj.Next(0, rndLayer.neurons.Count)).Value;

                    WeightHandler.removeAllConnections(rndHidden.ID, this);
                    rndLayer.neurons.Remove(rndHidden.ID);
                    remUseless = true;
                }
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron + chanceRandomFunction) / 100f) {
                if (layerManager.layerCount > 2) {
                    Layer rndLayer = layerManager.getRandomHiddenLayer(rndObj);
                    Neuron rndHidden = rndLayer.neurons.ElementAt(rndObj.Next(0, rndLayer.neurons.Count)).Value;

                    rndHidden.function = getRandomFunction(rndObj);
                }
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron + chanceRandomFunction + chanceAddRecurrentWeight) / 100f) {
                if (layerManager.layerCount > 2) {
                    Neuron rndAction = layerManager.actionLayer.neurons.ElementAt(rndObj.Next(0, layerManager.actionLayer.neurons.Count)).Value;
                    Layer rndHiddenLayer = layerManager.getRandomHiddenLayer(rndObj);
                    Neuron rndHidden = rndHiddenLayer.neurons.ElementAt(rndObj.Next(0, rndHiddenLayer.neurons.Count)).Value;

                    WeightHandler.addWeight(rndAction.ID, rndHidden.ID, UtilityClass.RandomWeight(rndObj), this);
                }
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron + chanceRandomFunction + chanceAddRecurrentWeight + chanceUpdateWeight) / 100f) {
                List<KeyValuePair<int, List<Connection>>> cons = new List<KeyValuePair<int, List<Connection>>>(connectionList);
                cons.AddRange(recurrentConnectionList);

                if (cons.Count != 0) {
                    KeyValuePair<int, List<Connection>> pair = cons.ElementAt(rndObj.Next(0, cons.Count));
                    Connection rndConnection = pair.Value.ElementAt(rndObj.Next(0, pair.Value.Count));
                    
                    float rndSign = rndObj.Next(0, 2) * 2 - 1;
                    WeightHandler.updateWeight(pair.Key, rndConnection.targetID, UtilityClass.Clamp(-4f, 4f, rndConnection.weight + rndSign * (float)rndObj.NextDouble()), this);
                }
            }
            
            if (remUseless && layerManager.layerCount > 2) {
                removeUselessHidden();
            }
        }

        public void removeUselessHidden() {

            List<int> allTargets = connectionList.SelectMany(x => x.Value).ToList().Select(x => x.targetID).ToList();

            for (int i = 1; i < layerManager.layerCount-1; i++) {
                foreach (Neuron hiddenNeuron in layerManager.allLayers[i].neurons.Values.ToList()) {
                    //if neuron is not source af any connection, remove
                    if (!connectionList.ContainsKey(hiddenNeuron.ID) || !allTargets.Contains(hiddenNeuron.ID)) {
                        layerManager.allLayers[i].neurons.Remove(hiddenNeuron.ID);
                        WeightHandler.removeAllConnections(hiddenNeuron.ID, this);
                    }
                }
            }

            removeUselessLayer();
        }

        private void removeUselessLayer() {
            int lastHiddenIndex = layerManager.layerCount - 2;
            for (int i = lastHiddenIndex; i != 0; i--) {
                if (layerManager.allLayers[i].neurons.Count == 0) {
                    layerManager.allLayers.RemoveAt(i);
                }
            }
        }

        public void calculateNetwork() {

            //reset all hidden
            foreach (Layer hLayer in layerManager.hiddenLayers) {
                foreach (Neuron neuron in hLayer.neurons.Values) {
                    neuron.value = 0f;
                }
            }

            //apply recurrent
            foreach (KeyValuePair<int, List<Connection>> pair in recurrentConnectionList) {
                Neuron currentSource = layerManager.getNeuron(pair.Key);
                foreach (Connection connection in pair.Value) {
                    layerManager.getNeuron(connection.targetID).value += currentSource.value * connection.weight;
                }
            }

            //reset action layer
            foreach (Neuron neuron in layerManager.actionLayer.neurons.Values) {
                neuron.value = 0f;
            }

            //calculate per neuron
            for (int i = 0; i < layerManager.layerCount; i++) {
                foreach (KeyValuePair<int, Neuron> sourceNeuron in layerManager.allLayers[i].neurons) {

                    //calc function value
                    sourceNeuron.Value.processValue();

                    //if connections from current neuron exists
                    if (connectionList.TryGetValue(sourceNeuron.Key, out List<Connection> cons)) {
                        foreach (Connection con in cons) {
                            layerManager.getNeuron(con.targetID, i+1).value += sourceNeuron.Value.value * con.weight;
                        }
                    }
                }
            }
        }

        private ActivationFunction getRandomFunction(Random rnd) {
            return (ActivationFunction)rnd.Next(0, Enum.GetValues(typeof(ActivationFunction)).Length);
        }
    }
}
