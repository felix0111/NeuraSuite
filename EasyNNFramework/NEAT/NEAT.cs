using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Random = System.Random;

namespace EasyNNFramework.NEAT {
    [Serializable]
    public class NEAT {

        public Neuron[] InputNeurons, HiddenNeurons, ActionNeurons;

        public Dictionary<int, Neuron> Neurons = new Dictionary<int, Neuron>();

        public Dictionary<int, Connection> Connections = new Dictionary<int, Connection>();
        public Dictionary<int, Connection> RecurrentConnections = new Dictionary<int, Connection>();

        private int _connectionIDCounter, _neuronIDCounter;

        private int Depth => ls.layerArray.Count;

        private LayerStructure ls;

        //creates new neural net with given input and action neurons
        //the given neurons are used as templates
        //only the activation function, ID and Type stays the same, everything else will be newly initialized
        public NEAT(Neuron[] templateIns, Neuron[] templateOuts) {

            //TODO check if IDs and neuron types of template neurons are OK

            foreach (var neuron in templateIns) {
                Neurons.Add(neuron.ID, new Neuron(neuron.ID, neuron.Function, neuron.Type));
            }

            foreach (var neuron in templateOuts) {
                Neurons.Add(neuron.ID, new Neuron(neuron.ID, neuron.Function, neuron.Type));
            }

            _neuronIDCounter = Neurons.Max(o => o.Key) + 1;

            RecalculateNeuronArrays();
            ls = new LayerStructure(this);
        }

        public NEAT(in NEAT neat) {
            
            foreach (var neuron in neat.Neurons.Values) {
                Neurons.Add(neuron.ID, neuron.Clone());
            }
            
            foreach (var connection in neat.Connections.Values) {
                Connections.Add(connection.ID, connection);
            }

            foreach (var connection in neat.RecurrentConnections.Values) {
                RecurrentConnections.Add(connection.ID, connection);
            }

            _neuronIDCounter = Neurons.Max(o => o.Key) + 1;
            if(Connections.Count != 0) _connectionIDCounter = Connections.Max(o => o.Key) + 1;

            RecalculateNeuronArrays();
            ls = new LayerStructure(this);
        }

        private void RecalculateNeuronArrays() {
            InputNeurons = Neurons.Values.Where(o => o.Type == NeuronType.Input || o.Type == NeuronType.Bias).ToArray();
            HiddenNeurons = Neurons.Values.Where(o => o.Type == NeuronType.Hidden).ToArray();
            ActionNeurons = Neurons.Values.Where(o => o.Type == NeuronType.Action).ToArray();
        }

        //chances must add up to 100
        public void Mutate(System.Random rndObj, float chanceAddWeight, float chanceRandomizeWeight, float chanceRemoveWeight, float chanceAddNeuron, float chanceRemoveNeuron, float chanceRandomFunction, float chanceAddRecurrentWeight, float chanceUpdateWeight, ActivationFunction hiddenActivationFunction) {
            List<Neuron> possibleStartNeurons = new List<Neuron>(InputNeurons);
            List<Neuron> possibleEndNeurons = new List<Neuron>(ActionNeurons);

            //add hidden neurons if available
            if (HiddenNeurons.Length > 0) {
                possibleStartNeurons.AddRange(HiddenNeurons);
                possibleEndNeurons.AddRange(HiddenNeurons);
            }

            restart:
            //choose random neurons
            int rndStart = possibleStartNeurons[rndObj.Next(0, possibleStartNeurons.Count)].ID;
            int rndEnd = possibleEndNeurons[rndObj.Next(0, possibleEndNeurons.Count)].ID;

            if (rndStart == rndEnd) goto restart;
            
            float rndChance = (float)rndObj.NextDouble();
            if (rndChance <= chanceAddWeight / 100f) {
                if (CheckRecurrent(rndStart, rndEnd)) (rndStart, rndEnd) = (rndEnd, rndStart);
                AddConnection(rndStart, rndEnd, UtilityClass.RandomWeight(rndObj));
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight) / 100f) {
                if (Connections.Count == 0) return;
                Connection con = RecurrentConnections.Count != 0 ? RandomConnection(rndObj.NextDouble() > 0.5D, rndObj) : RandomConnection(false, rndObj);
                ChangeWeight(con.ID, UtilityClass.RandomWeight(rndObj));
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight) / 100f) {
                if(Connections.Count == 0) return;
                Connection con = RecurrentConnections.Count != 0 ? RandomConnection(rndObj.NextDouble() > 0.5D, rndObj) : RandomConnection(false, rndObj);
                RemoveConnection(con.ID);
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron) / 100f) {
                if (Connections.Count == 0) return;
                Connection con = RandomConnection(false, rndObj);
                AddNeuron(con.ID, hiddenActivationFunction);
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron) / 100f) {
                if (HiddenNeurons.Length == 0) return;
                RemoveNeuron(HiddenNeurons[rndObj.Next(0, HiddenNeurons.Length)].ID);
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron + chanceRandomFunction) / 100f) {
                if (HiddenNeurons.Length == 0) return;
                HiddenNeurons[rndObj.Next(0, HiddenNeurons.Length)].Function = GetRandomFunction(rndObj);
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron + chanceRandomFunction + chanceAddRecurrentWeight) / 100f) {
                if (HiddenNeurons.Length == 0) return;

                if (GetNeuronType(rndStart) == NeuronType.Input) rndStart = HiddenNeurons[rndObj.Next(0, HiddenNeurons.Length)].ID;
                if (rndStart == rndEnd) return;

                if (GetNeuronType(rndEnd) != NeuronType.Action && !CheckRecurrent(rndEnd, rndStart)) (rndEnd, rndStart) = (rndStart, rndEnd);
                
                AddConnection(rndEnd, rndStart, UtilityClass.RandomWeight(rndObj));
            } else if (rndChance <= (chanceAddWeight + chanceRandomizeWeight + chanceRemoveWeight + chanceAddNeuron + chanceRemoveNeuron + chanceRandomFunction + chanceAddRecurrentWeight + chanceUpdateWeight) / 100f) {
                if(Connections.Count == 0) return;
                Connection con = RecurrentConnections.Count != 0 ? RandomConnection(rndObj.NextDouble() > 0.5D, rndObj) : RandomConnection(false, rndObj);

                float rndSign = rndObj.Next(0, 2) * 2 - 1;
                ChangeWeight(con.ID, UtilityClass.Clamp(-4f, 4f, con.Weight + rndSign * (float)rndObj.NextDouble()));
            }

            RecalculateStructure();
        }

        public void RecalculateStructure() {
            
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

            foreach (Neuron n in HiddenNeurons) {

                if (n.IncommingConnections.Count == 0 || n.OutgoingConnections.Count == 0) {
                    useless.Add(n);
                }
            }

            return useless;
        }

        public void AddNeuron(int connectionID, ActivationFunction function) {
            if (RecurrentConnections.ContainsKey(connectionID)) throw new Exception("Can't add neuron between recurent connection!");

            Connection con = Connections[connectionID];
            Neurons.Add(_neuronIDCounter, new Neuron(_neuronIDCounter, function, NeuronType.Hidden));
            RecalculateNeuronArrays();

            RemoveConnection(con.ID);
            AddConnection(con.SourceID, _neuronIDCounter, con.Weight);
            AddConnection(_neuronIDCounter, con.TargetID, 1f);

            _neuronIDCounter++;
        }

        public void RemoveNeuron(int ID) {
            //remove all connections that use this neuron id
            RemoveDependingConnections(ID, true);

            //remove actual neuron from array
            Neurons.Remove(ID);
            RecalculateNeuronArrays();
        }

        public void AddConnection(int sourceID, int targetID, float weight) {
            if (ExistsConnection(sourceID, targetID)) return;
            if (CheckRecurrent(sourceID, targetID)) {
                RecurrentConnections.Add(_connectionIDCounter, new Connection(_connectionIDCounter, sourceID, targetID, weight));
                _connectionIDCounter++;
            } else {
                Connections.Add(_connectionIDCounter, new Connection(_connectionIDCounter, sourceID, targetID, weight));
                Neurons[sourceID].OutgoingConnections.Add(_connectionIDCounter);
                Neurons[targetID].IncommingConnections.Add(_connectionIDCounter);
                _connectionIDCounter++;
            }
        }

        public void ChangeWeight(int connectionID, float newWeight) {
            if (RecurrentConnections.TryGetValue(connectionID, out Connection val)) {
                val.Weight = newWeight;
                RecurrentConnections[connectionID] = val;
            } else if(Connections.TryGetValue(connectionID, out Connection val2)) {
                val2.Weight = newWeight;
                Connections[connectionID] = val2;
            } else {
                //no connection found
            }
        }

        public void RemoveConnection(int connectionID) {
            if (Connections.TryGetValue(connectionID, out Connection val)) {
                Connections.Remove(connectionID);
                Neurons[val.SourceID].OutgoingConnections.Remove(connectionID);
                Neurons[val.TargetID].IncommingConnections.Remove(connectionID);
            }

            RecurrentConnections.Remove(connectionID);
        }

        public void RemoveDependingConnections(int neuronID, bool inclRecurrent) {
            foreach (var connection in Connections.Values.Where(o => o.SourceID == neuronID || o.TargetID == neuronID).ToList()) {
                RemoveConnection(connection.ID);
            }

            if (inclRecurrent) {
                foreach (var connection in RecurrentConnections.Values.Where(o => o.SourceID == neuronID || o.TargetID == neuronID).ToList()) {
                    RemoveConnection(connection.ID);
                }
            }
        }

        public bool CheckRecurrent(int sourceID, int searchingFor) {
            if (GetNeuronType(sourceID) == NeuronType.Action) return true;  //connection starting at action neuron
            if (GetNeuronType(sourceID) == NeuronType.Input || GetNeuronType(sourceID) == NeuronType.Bias) return false;  //connection starting at input neuron
            if (GetNeuronType(searchingFor) == NeuronType.Action) return false; //connection ending at action neuron

            //check if searching for neuron exists in incomming connections
            foreach (int connectionID in Neurons[sourceID].IncommingConnections) {
                Connection con = Connections[connectionID];
                if (con.SourceID == searchingFor) return true;

                //if connection source is hidden, search for targetID in incomming connections
                if (CheckRecurrent(con.SourceID, searchingFor)) return true;
            }

            return false;
        }

        public bool ExistsConnection(int sourceID, int targetID) {
            Connection buffer;

            for (int i = 0; i < RecurrentConnections.Count; i++) {
                buffer = RecurrentConnections.Values.ElementAt(i);
                if (buffer.SourceID == sourceID && buffer.TargetID == targetID) return true;
            }

            for (int i = 0; i < Connections.Count; i++) {
                buffer = Connections.Values.ElementAt(i);
                if (buffer.SourceID == sourceID && buffer.TargetID == targetID) return true;
            }

            return false;
        }

        public Connection RandomConnection(bool recurrent, Random rnd) {
            return recurrent ? RecurrentConnections.Values.ElementAt(rnd.Next(0, RecurrentConnections.Count)) : Connections.Values.ElementAt(rnd.Next(0, Connections.Count));
        }

        public void ResetNetwork() {
            //update neuron values
            for (int i = 0; i < InputNeurons.Length; i++) {
                InputNeurons[i].ResetState();
                InputNeurons[i].ResetState();
            }
            for (int i = 0; i < HiddenNeurons.Length; i++) {
                HiddenNeurons[i].ResetState();
                HiddenNeurons[i].ResetState();
            }
            for (int i = 0; i < ActionNeurons.Length; i++) {
                ActionNeurons[i].ResetState();
                ActionNeurons[i].ResetState();
            }
        }

        public void CalculateNetwork(in float[] inputNeuronValues, ref float[] actionNeuronValues) {

            //update neuron values
            for (int i = 0; i < InputNeurons.Length; i++) {
                InputNeurons[i].ResetState();
                InputNeurons[i].Input(inputNeuronValues[i]);
            }
            for (int i = 0; i < HiddenNeurons.Length; i++) {
                HiddenNeurons[i].ResetState();
            }
            for (int i = 0; i < ActionNeurons.Length; i++) {
                ActionNeurons[i].ResetState();
            }

            //apply recurrent
            //this is done by using the last value of the corresponding neurons
            foreach (Connection con in RecurrentConnections.Values) {
                Neurons[con.TargetID].Input(con.Weight * Neurons[con.SourceID].LastValue);
            }

            //calculate all neurons layer by layer
            //first sum all values from incomming connections, then activate neuron, goto next layer
            for (int layerIndex = 0; layerIndex < ls.layerArray.Count; layerIndex++) {
                for (int i = 0; i < ls.layerArray[layerIndex].Length; i++) {
                    Neuron target = Neurons[ls.layerArray[layerIndex][i]];

                    for(int j = 0; j < target.IncommingConnections.Count; j++) {
                        Connection con = Connections[target.IncommingConnections[j]];
                        Neuron src = Neurons[con.SourceID];

                        //shouldnt happen
                        if (!src.Activated) throw new Exception("A neuron was not activated in the feed forward process!");

                        target.Input(con.Weight * src.Value);
                    }

                    target.Activate();
                }
            }

            //set output values
            for (int i = 0; i < ActionNeurons.Length; i++) {
                actionNeuronValues[i] = ActionNeurons[i].Value;
            }
        }

        private static ActivationFunction GetRandomFunction(Random rnd) {
            return (ActivationFunction)rnd.Next(0, Enum.GetValues(typeof(ActivationFunction)).Length);
        }

        public NeuronType GetNeuronType(int ID) {
            return Neurons[ID].Type;
        }
        
    }

    [Serializable]
    public struct LayerStructure {

        public List<int[]> layerArray;
        private Dictionary<int, int> neuronLayerDict;

        public LayerStructure(in NEAT neat) {
            layerArray = new List<int[]>();
            neuronLayerDict = new Dictionary<int, int>(neat.Neurons.Count); //key = neuron id, value = layer

            //calc layer for all neurons
            foreach (Neuron n in neat.Neurons.Values) {
                GetLayer(n.ID, neat);
            }
            
            //add layers
            for (int i = 1; i <= neuronLayerDict.Values.Max(); i++) {
                var allNeuronsInLayerI = neuronLayerDict.Where(o => o.Value == i).Select(o => o.Key).ToArray();
                layerArray.Add(allNeuronsInLayerI);
            }
        }

        public int GetLayer(int neuronID, in NEAT neat) {
            if (neuronLayerDict.TryGetValue(neuronID, out int v)) return v;

            int highestLayer = 1;
            for (int i = 0; i < neat.Neurons[neuronID].IncommingConnections.Count; i++) {
                Connection con = neat.Connections[neat.Neurons[neuronID].IncommingConnections[i]];
                int l = GetLayer(con.SourceID, neat);
                if (l > highestLayer) highestLayer = l;
            }

            neuronLayerDict.Add(neuronID, highestLayer + 1);
            return highestLayer + 1;
        }
    }

    [Serializable]
    public struct Connection : IEquatable<Connection> {

        public float Weight;
        public int TargetID, SourceID;

        public int ID { get; private set; }

        public Connection(int connectionID, int sourceID, int targetID, float weight) {
            ID = connectionID;
            SourceID = sourceID;
            TargetID = targetID;
            Weight = weight;
        }

        public override bool Equals(object obj) => obj is Connection n && Equals(n);

        public static bool operator ==(Connection lf, Connection ri) => lf.Equals(ri);

        public static bool operator !=(Connection lf, Connection ri) => !(lf == ri);

        public override int GetHashCode() => ID.GetHashCode();

        public bool Equals(Connection obj) => obj.ID == ID;

        public Connection Clone() {
            return this;
        }
    }
}
