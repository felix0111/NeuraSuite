using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Random = System.Random;

namespace EasyNNFramework.NEAT {
    [Serializable]
    public class NEAT {

        public readonly Neuron[] ROinputNeurons, ROactionNeurons;

        public Neuron[] inputNeurons;
        public Neuron[] hiddenNeurons;
        public Neuron[] actionNeurons;
        
        public Connection[] connectionList;
        public Connection[] recurrentConnectionList;

        public int IDCounter;

        private LayerStructure ls;

        public NEAT(Neuron[] inputs, Neuron[] action) {
            ROinputNeurons = inputs.CopyNeuronArray();
            ROactionNeurons = action.CopyNeuronArray();

            inputNeurons = inputs.CopyNeuronArray();
            hiddenNeurons = Array.Empty<Neuron>();
            actionNeurons = action.CopyNeuronArray();

            connectionList = Array.Empty<Connection>();
            recurrentConnectionList = Array.Empty<Connection>();
            IDCounter = inputs.Length + action.Length;

            ls = new LayerStructure(this);
        }

        public NEAT(in NEAT neat) {
            ROinputNeurons = neat.ROinputNeurons.CopyNeuronArray();
            ROactionNeurons = neat.ROactionNeurons.CopyNeuronArray();

            inputNeurons = neat.inputNeurons.CopyNeuronArray();
            hiddenNeurons = neat.hiddenNeurons.CopyNeuronArray();
            actionNeurons = neat.actionNeurons.CopyNeuronArray();

            connectionList = (Connection[]) neat.connectionList.Clone();
            recurrentConnectionList = (Connection[]) neat.recurrentConnectionList.Clone();

            IDCounter = neat.IDCounter;

            ls = new LayerStructure(this);
        }

        //chances must add up to 100
        public void Mutate(System.Random rndObj, float chanceAddWeight, float chanceRandomizeWeight, float chanceRemoveWeight, float chanceAddNeuron, float chanceRemoveNeuron, float chanceRandomFunction, float chanceAddRecurrentWeight, float chanceUpdateWeight, ActivationFunction hiddenActivationFunction) {
            List<Neuron> possibleStartNeurons = new List<Neuron>(ROinputNeurons);
            List<Neuron> possibleEndNeurons = new List<Neuron>(ROactionNeurons);

            //add hidden neurons if available
            if (hiddenNeurons.Length > 0) {
                possibleStartNeurons.AddRange(hiddenNeurons);
                possibleEndNeurons.AddRange(hiddenNeurons);
            }

            restart:
            //choose random neurons
            int rndStart = possibleStartNeurons[rndObj.Next(0, possibleStartNeurons.Count)].ID;
            int rndEnd = possibleEndNeurons[rndObj.Next(0, possibleEndNeurons.Count)].ID;

            if (rndStart == rndEnd) goto restart;
            
            float rndChance = (float)rndObj.NextDouble();
            if (rndChance <= chanceAddWeight / 100f) {
                if (isRecurrentConnection(rndStart, rndEnd)) (rndStart, rndEnd) = (rndEnd, rndStart);
                AddWeight(rndStart, rndEnd, UtilityClass.RandomWeight(rndObj));
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight) / 100f) {
                if (connectionList.Length == 0) return;
                Connection con = recurrentConnectionList.Length != 0 ? RandomConnection(rndObj.NextDouble() > 0.5D, rndObj) : RandomConnection(false, rndObj);
                ChangeWeight(con.sourceID, con.targetID, UtilityClass.RandomWeight(rndObj));
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight) / 100f) {
                if(connectionList.Length == 0) return;
                Connection con = recurrentConnectionList.Length != 0 ? RandomConnection(rndObj.NextDouble() > 0.5D, rndObj) : RandomConnection(false, rndObj);
                RemoveWeight(con.sourceID, con.targetID);
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron) / 100f) {
                if (connectionList.Length == 0) return;
                Connection con = RandomConnection(false, rndObj);
                AddNeuron(con.sourceID, con.targetID, hiddenActivationFunction);
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron) / 100f) {
                if (hiddenNeurons.Length == 0) return;
                RemoveNeuron(hiddenNeurons[rndObj.Next(0, hiddenNeurons.Length)].ID);
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron + chanceRandomFunction) / 100f) {
                if (hiddenNeurons.Length == 0) return;
                hiddenNeurons[rndObj.Next(0, hiddenNeurons.Length)].function = GetRandomFunction(rndObj);
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron + chanceRandomFunction + chanceAddRecurrentWeight) / 100f) {
                if (hiddenNeurons.Length == 0) return;

                if (GetNeuronType(rndStart) == NeuronType.Input) rndStart = hiddenNeurons[rndObj.Next(0, hiddenNeurons.Length)].ID;
                if (rndStart == rndEnd) return;

                if (GetNeuronType(rndEnd) != NeuronType.Action && !isRecurrentConnection(rndEnd, rndStart)) (rndEnd, rndStart) = (rndStart, rndEnd);
                
                AddWeight(rndEnd, rndStart, UtilityClass.RandomWeight(rndObj));
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron + chanceRandomFunction + chanceAddRecurrentWeight + chanceUpdateWeight) / 100f) {
                if(connectionList.Length == 0) return;
                Connection con = recurrentConnectionList.Length != 0 ? RandomConnection(rndObj.NextDouble() > 0.5D, rndObj) : RandomConnection(false, rndObj);

                float rndSign = rndObj.Next(0, 2) * 2 - 1;
                ChangeWeight(con.sourceID, con.targetID, UtilityClass.Clamp(-4f, 4f, con.weight + rndSign * (float)rndObj.NextDouble()));
            }

            recalculateStructure();
        }

        public void recalculateStructure() {
            
            //rm useless hidden
            List<Neuron> useless = UselessHidden();
            do {
                foreach (var n in useless) {
                    RemoveNeuron(n.ID);
                }

                useless = UselessHidden();
            } while (useless.Count != 0);

            ls = new LayerStructure(this);
        }

        private List<Neuron> UselessHidden() {
            List<Neuron> useless = new List<Neuron>();

            foreach (Neuron n in hiddenNeurons) {

                if (n.incommingConnections.Count == 0 || n.outgoingConnections.Count == 0) {
                    useless.Add(n);
                }
            }

            return useless;
        }

        public void AddNeuron(int sourceID, int targetID, ActivationFunction function) {
            if (isRecurrentConnection(sourceID, targetID)) throw new Exception("Can't add neuron between recurent connection!");

            Connection con = GetConnection(sourceID, targetID);
            hiddenNeurons = hiddenNeurons.Concat(new []{new Neuron(IDCounter, function, NeuronType.Hidden)}).ToArray();

            RemoveWeight(con.sourceID, con.targetID);
            AddWeight(sourceID, IDCounter, con.weight);
            AddWeight(IDCounter, con.targetID, 1f);

            IDCounter++;
        }

        public void RemoveNeuron(int ID) {
            //remove all connections that use this neuron id
            RemoveDependingConnections(ID);
            recurrentConnectionList = recurrentConnectionList.Where(o => !(o.sourceID == ID || o.targetID == ID)).ToArray();

            //remove actual neuron from array
            hiddenNeurons = hiddenNeurons.Where(o => o.ID != ID).ToArray();
        }

        public void AddWeight(int sourceID, int targetID, float weight) {
            if (isRecurrentConnection(sourceID, targetID)) {
                if (Array.Exists(recurrentConnectionList, o => o.sourceID == sourceID && o.targetID == targetID)) return;
                recurrentConnectionList = recurrentConnectionList.Concat( new []{new Connection(sourceID, targetID, weight)} ).ToArray();
            } else {
                if (Array.Exists(connectionList, o => o.sourceID == sourceID && o.targetID == targetID)) return;
                connectionList = connectionList.Concat(new []{new Connection(sourceID, targetID, weight)} ).ToArray();
                GetNeuronRef(sourceID).outgoingConnections.Add(targetID);
                GetNeuronRef(targetID).incommingConnections.Add(sourceID);
            }
        }

        public void ChangeWeight(int sourceID, int targetID, float newWeight) {
            if (isRecurrentConnection(sourceID, targetID)) {
                int index = Array.FindIndex(recurrentConnectionList, o => o.sourceID == sourceID && o.targetID == targetID);
                if (index == -1) return;
                recurrentConnectionList[index] = new Connection(sourceID, targetID, newWeight);
            } else {
                int index = Array.FindIndex(connectionList, o => o.sourceID == sourceID && o.targetID == targetID);
                if(index == -1) return;
                connectionList[index] = new Connection(sourceID, targetID, newWeight);
            }
        }

        public void RemoveWeight(int sourceID, int targetID) {
            if (isRecurrentConnection(sourceID, targetID)) {
                recurrentConnectionList = recurrentConnectionList.Where(o => !(o.sourceID == sourceID && o.targetID == targetID)).ToArray();
            } else {
                connectionList = connectionList.Where(o => !(o.sourceID == sourceID && o.targetID == targetID)).ToArray();
                GetNeuronRef(sourceID).outgoingConnections.Remove(targetID);
                GetNeuronRef(targetID).incommingConnections.Remove(sourceID);
            }
        }

        public void RemoveDependingConnections(int neuronID) {
            foreach (var connection in connectionList.Where(o => o.sourceID == neuronID || o.targetID == neuronID)) {
                RemoveWeight(connection.sourceID, connection.targetID);
            }
        }

        public bool isRecurrentConnection(int sourceID, int searchingFor) {
            if (GetNeuronType(sourceID) == NeuronType.Action) return true;  //connection starting at action neuron
            if (GetNeuronType(sourceID) == NeuronType.Input) return false;  //connection starting at input neuron
            if (GetNeuronType(searchingFor) == NeuronType.Action) return false; //connection ending at action neuron

            //check if searching for neuron exists in incomming connections
            foreach (int connectionSource in GetNeuronRef(sourceID).incommingConnections) {
                if (connectionSource == searchingFor) return true;

                //if connection source is hidden, search for targetID in incomming connections
                if (GetNeuronType(connectionSource) != NeuronType.Input) {
                    if (isRecurrentConnection(connectionSource, searchingFor)) return true;
                }
            }

            return false;
        }

        public Connection GetConnection(int sourceID, int targetID) {

            for (int i = 0; i < connectionList.Length; i++) {
                if (connectionList[i].sourceID == sourceID && connectionList[i].targetID == targetID) return connectionList[i];
            }

            throw new Exception("No Connection with source ID: " + sourceID + " and target ID " + targetID + " found!");
        }

        public Connection GetRecurrentConnection(int sourceID, int targetID) {
            for (int i = 0; i < recurrentConnectionList.Length; i++) {
                if (recurrentConnectionList[i].sourceID == sourceID && recurrentConnectionList[i].targetID == targetID) return recurrentConnectionList[i];
            }

            throw new Exception("No Recurrent connection with source ID: " + sourceID + " and target ID " + targetID + " found!");
        }

        public Connection RandomConnection(bool recurrent, Random rnd) {
            return recurrent ? recurrentConnectionList[rnd.Next(0, recurrentConnectionList.Length)] : connectionList[rnd.Next(0, connectionList.Length)];
        }

        private List<int> neuronsToCalcBuffer = new List<int>();
        public void CalculateNetwork(in float[] inputNeuronValues, ref float[] actionNeuronValues) {

            //update neuron values
            for (int i = 0; i < inputNeurons.Length; i++) {
                inputNeurons[i].lastValue = inputNeurons[i].value;
                inputNeurons[i].value = inputNeuronValues[i];
            }
            for (int i = 0; i < hiddenNeurons.Length; i++) {
                hiddenNeurons[i].lastValue = hiddenNeurons[i].value;
                hiddenNeurons[i].value = 0f;
            }
            for (int i = 0; i < actionNeurons.Length; i++) {
                actionNeurons[i].lastValue = actionNeurons[i].value;
                actionNeurons[i].value = 0f;
            }

            //apply recurrent
            foreach (Connection con in recurrentConnectionList) {
                GetNeuronRef(con.targetID).value += con.weight * GetNeuronRef(con.sourceID).lastValue;
            }

            //feed forward input neurons
            for (int i = 0; i < inputNeurons.Length; i++) {
                for (int j = 0; j < inputNeurons[i].outgoingConnections.Count; j++) {
                    int targetID = inputNeurons[i].outgoingConnections[j];
                    int targetIndex = GetHiddenIndex(targetID);

                    if (targetIndex == -1) {
                        actionNeurons[targetID-ROinputNeurons.Length].value += GetConnection(i, targetID).weight * inputNeurons[i].value;
                    } else {
                        hiddenNeurons[targetIndex].value += GetConnection(i, targetID).weight * inputNeurons[i].value;
                    }
                }
            }

            //calculate hidden
            for (int layerIndex = 1; layerIndex < ls.layerArray.Count-1; layerIndex++) {
                for (int i = 0; i < ls.layerArray[layerIndex].Length; i++) {
                    int hiddenIndex = GetHiddenIndex(ls.layerArray[layerIndex][i]);

                    hiddenNeurons[hiddenIndex].processValue();
                    for(int j = 0; j < hiddenNeurons[hiddenIndex].outgoingConnections.Count; j++) {
                        int targetID = hiddenNeurons[hiddenIndex].outgoingConnections[j];
                        int targetIndex = GetHiddenIndex(targetID);

                        if (targetIndex == -1) {
                            actionNeurons[targetID - ROinputNeurons.Length].value += GetConnection(hiddenNeurons[hiddenIndex].ID, targetID).weight * hiddenNeurons[hiddenIndex].value;
                        } else {
                            hiddenNeurons[targetIndex].value += GetConnection(hiddenNeurons[hiddenIndex].ID, targetID).weight * hiddenNeurons[hiddenIndex].value;
                        }
                    }
                }
            }

            //calculate action
            for (int i = 0; i < ROactionNeurons.Length; i++) {
                actionNeurons[i].processValue();
                actionNeuronValues[i] = actionNeurons[i].value;
            }
        }

        private static ActivationFunction GetRandomFunction(Random rnd) {
            return (ActivationFunction)rnd.Next(0, Enum.GetValues(typeof(ActivationFunction)).Length);
        }

        public int GetHiddenIndex(int ID) {
            for (int i = 0; i < hiddenNeurons.Length; i++) {
                if (hiddenNeurons[i].ID == ID) return i;
            }

            return -1;
        }

        public NeuronType GetNeuronType(int ID) {
            if (ID < ROinputNeurons.Length) return NeuronType.Input;
            if (ID < ROinputNeurons.Length + ROactionNeurons.Length) return NeuronType.Action;
            return NeuronType.Hidden;
        }

        public ref Neuron GetNeuronRef(int ID) {
            switch (GetNeuronType(ID)) {
                case NeuronType.Input: return ref inputNeurons[ID];
                case NeuronType.Hidden: return ref hiddenNeurons[GetHiddenIndex(ID)];
                case NeuronType.Action: return ref actionNeurons[ID - inputNeurons.Length];
                default: throw new Exception("Neuron with ID " + ID + " not found.");
            }
        }
    }

    [Serializable]
    public struct LayerStructure {

        public List<int[]> layerArray;
        private Dictionary<int, int> neuronLayerDict;

        public LayerStructure(in NEAT neat) {
            layerArray = new List<int[]>();
            neuronLayerDict = new Dictionary<int, int>(neat.inputNeurons.Length + neat.actionNeurons.Length + neat.hiddenNeurons.Length);

            //set layer for all input neurons
            for (int currentID = 0; currentID < neat.inputNeurons.Length; currentID++) {
                if (currentID < neat.inputNeurons.Length) {
                    neuronLayerDict.Add(currentID, 1);
                }
            }

            //calc layer for all neurons
            foreach (Neuron n in neat.hiddenNeurons) {
                GetLayer(n.ID, neat);
            }

            //add input layer
            layerArray.Add(neat.ROinputNeurons.Select(o => o.ID).ToArray());

            //add hidden layers
            for (int i = 2; i <= neuronLayerDict.Values.Max(); i++) {
                var allNeuronsInLayerI = neuronLayerDict.Where(o => o.Value == i).Select(o => o.Key).ToArray();
                layerArray.Add(allNeuronsInLayerI);
            }

            //add action layer
            layerArray.Add(neat.ROactionNeurons.Select(o => o.ID).ToArray());

        }

        public int GetLayer(int neuronID, in NEAT neat) {
            if (neuronLayerDict.TryGetValue(neuronID, out int v)) return v;

            int highestLayer = 1;
            for (int i = 0; i < neat.GetNeuronRef(neuronID).incommingConnections.Count; i++) {
                int l = GetLayer(neat.GetNeuronRef(neuronID).incommingConnections[i], neat);
                if (l > highestLayer) highestLayer = l;
            }

            neuronLayerDict.Add(neuronID, highestLayer + 1);
            return highestLayer + 1;
        }
    }

    [Serializable]
    public struct Connection : IEquatable<Connection> {
        public float weight;
        public int targetID, sourceID;

        public Connection(int sourceID, int targetID, float weight) {
            this.sourceID = sourceID;
            this.targetID = targetID;
            this.weight = weight;
        }

        public override bool Equals(object obj) => obj is Connection n && Equals(n);

        public static bool operator ==(Connection lf, Connection ri) => lf.Equals(ri);

        public static bool operator !=(Connection lf, Connection ri) => !(lf == ri);

        public override int GetHashCode() => (sourceID.GetHashCode() + targetID.GetHashCode() + weight.GetHashCode()).GetHashCode();

        public bool Equals(Connection obj) => obj.sourceID == sourceID && obj.targetID == targetID;
    }
}
