using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace EasyNNFramework.NEAT {

    [Serializable]
    public class Network {

        //all these arrays are automatically sorted by ID
        public Neuron[] InputNeurons { get; private set; }
        public Neuron[] HiddenNeurons { get; private set; }
        public Neuron[] ActionNeurons { get; private set; }

        public Dictionary<int, Neuron> Neurons = new Dictionary<int, Neuron>();

        public Dictionary<int, Connection> Connections = new Dictionary<int, Connection>();
        public Dictionary<int, Connection> RecurrentConnections = new Dictionary<int, Connection>();

        //value at position x corresponds to the neuron in the specific array at position x
        public float[] InputValues, OutputValues;

        protected int _neuronIDCounter;

        public readonly int NetworkID;

        public int SpeciesID;

        public float Fitness { get; private set; }

        private int Depth => _ls.layerArray.Count;

        private LayerStructure _ls;

        //creates new neural net with given input and action neurons
        //the given neurons are used as templates
        //only the activation function, ID and Type stays the same, everything else will be newly initialized
        public Network(int networkID, Neuron[] templateIns, Neuron[] templateOuts) {

            NetworkID = networkID;

            //TODO check if IDs and neuron types of template neurons are OK
            foreach (var neuron in templateIns) {
                Neurons.Add(neuron.ID, new Neuron(neuron.ID, neuron.Function, neuron.Type));
            }

            foreach (var neuron in templateOuts) {
                Neurons.Add(neuron.ID, new Neuron(neuron.ID, neuron.Function, neuron.Type));
            }

            _neuronIDCounter = Neurons.Max(o => o.Key) + 1;

            RecalculateNeuronArrays();
            _ls = new LayerStructure(this);

            InputValues = new float[InputNeurons.Length];
            OutputValues = new float[ActionNeurons.Length];
        }

        //basically copies a network
        public Network(int networkID, in Network templateNetwork) {

            NetworkID = networkID;

            foreach (var neuron in templateNetwork.Neurons.Values) {
                Neurons.Add(neuron.ID, neuron.Clone());
            }

            foreach (var connection in templateNetwork.Connections.Values) {
                Connections.Add(connection.InnovationID, connection);
            }

            foreach (var connection in templateNetwork.RecurrentConnections.Values) {
                RecurrentConnections.Add(connection.InnovationID, connection);
            }

            _neuronIDCounter = templateNetwork._neuronIDCounter;

            RecalculateNeuronArrays();
            _ls = new LayerStructure(this);

            Fitness = templateNetwork.Fitness;

            InputValues = new float[InputNeurons.Length];
            OutputValues = new float[ActionNeurons.Length];
        }

        private void RecalculateNeuronArrays() {
            InputNeurons = Neurons.Values.Where(o => o.Type == NeuronType.Input || o.Type == NeuronType.Bias).ToArray();
            HiddenNeurons = Neurons.Values.Where(o => o.Type == NeuronType.Hidden).ToArray();
            ActionNeurons = Neurons.Values.Where(o => o.Type == NeuronType.Action).ToArray();
        }

        public void RecalculateStructure() {

            //rm useless hidden
            List<Neuron> useless = this.GetUselessHidden();
            do {
                foreach (var n in useless) {
                    RemoveNeuron(n.ID);
                }

                useless = this.GetUselessHidden();
            } while (useless.Count != 0);

            _ls = new LayerStructure(this);
        }
        
        //the neat object is needed for creating the innovation IDs
        public Neuron AddNeuron(Neat neat, int connectionID, ActivationFunction function) {
            if (RecurrentConnections.ContainsKey(connectionID)) throw new Exception("Can't add neuron between recurent connection!");
            if (!Connections.ContainsKey(connectionID)) throw new Exception("Connection is does not exist!");

            Connection con = Connections[connectionID];
            Neuron newNeuron = new Neuron(_neuronIDCounter, function, NeuronType.Hidden);
            Neurons.Add(newNeuron.ID, newNeuron);
            RecalculateNeuronArrays();

            RemoveConnection(con.InnovationID);
            AddConnection(neat.NewInnovation(con.SourceID, newNeuron.ID), con.SourceID, newNeuron.ID, con.Weight);
            AddConnection(neat.NewInnovation(newNeuron.ID, con.TargetID), newNeuron.ID, con.TargetID, 1f);

            _neuronIDCounter++;
            return newNeuron;
        }

        public void RemoveNeuron(int ID) {
            if (!Neurons.ContainsKey(ID)) throw new Exception("Neuron does not exist!");

            //remove all connections that use this neuron id
            RemoveDependingConnections(ID);

            //remove actual neuron from array
            Neurons.Remove(ID);
            RecalculateNeuronArrays();
        }
        
        //returns the created connection
        //NOT A REFERENCE, just a value type!
        //if connection is already existing, returns connection with values -1
        public Connection AddConnection(int innovID, int sourceID, int targetID, float weight) {
            //probably not necessary to check by neuron IDs AND innovation ID, one should be enough
            if (this.ExistsConnection(sourceID, targetID) || this.ExistsConnection(innovID)) return new Connection(-1, -1, -1, -1);

            Connection newConnection = new Connection(innovID, sourceID, targetID, weight);

            //recurrent connections are handled differently
            if (this.CheckRecurrent(sourceID, targetID)) {
                RecurrentConnections.Add(newConnection.InnovationID, newConnection);
            } else {
                Connections.Add(newConnection.InnovationID, newConnection);
                Neurons[sourceID].OutgoingConnections.Add(newConnection.InnovationID);
                Neurons[targetID].IncommingConnections.Add(newConnection.InnovationID);
            }

            return newConnection;
        }

        public void ChangeWeight(int connectionID, float newWeight) {
            if (RecurrentConnections.TryGetValue(connectionID, out Connection val)) {
                val.Weight = newWeight;
                RecurrentConnections[connectionID] = val;
            } else if (Connections.TryGetValue(connectionID, out Connection val2)) {
                val2.Weight = newWeight;
                Connections[connectionID] = val2;
            } else {
                throw new Exception("Connection with innovation ID " + connectionID + " was not found!");
            }
        }

        public void RemoveConnection(int connectionID) {
            if (Connections.TryGetValue(connectionID, out Connection val)) {
                Connections.Remove(connectionID);
                Neurons[val.SourceID].OutgoingConnections.Remove(connectionID);
                Neurons[val.TargetID].IncommingConnections.Remove(connectionID);
            } else if(!RecurrentConnections.Remove(connectionID)) {
                throw new Exception("Connection with innovation ID " + connectionID + " was not found!");
            }
        }

        //removes all connections and connection information in other neurons which depend on a specific neuron
        public void RemoveDependingConnections(int neuronID) {
            foreach (var connection in Connections.ToList()) {
                if (connection.Value.SourceID == neuronID || connection.Value.TargetID == neuronID) RemoveConnection(connection.Value.InnovationID);
            }

            foreach (var connection in RecurrentConnections.ToList()) {
                if (connection.Value.SourceID == neuronID || connection.Value.TargetID == neuronID) RemoveConnection(connection.Value.InnovationID);
            }
        }

        //resets the state of all neurons and their internal values (used for recurrent connections)
        public void ResetNetwork() {
            //reset neuron values
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
            ResetFitness();
        }

        public void CalculateNetwork() {

            //update neuron values
            for (int i = 0; i < InputNeurons.Length; i++) {
                InputNeurons[i].ResetState();
                InputNeurons[i].Input(InputValues[i]);
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
            for (int layerIndex = 0; layerIndex < _ls.layerArray.Count; layerIndex++) {
                //for each neuron id in layer
                for (int i = 0; i < _ls.layerArray[layerIndex].Length; i++) {
                    Neuron target = Neurons[_ls.layerArray[layerIndex][i]];

                    for (int j = 0; j < target.IncommingConnections.Count; j++) {
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
                OutputValues[i] = ActionNeurons[i].Value;
            }
        }

        //TODO function still work in progress
        public void CalculateStep(in float[] inputNeuronValues, ref float[] actionNeuronValues) {
            while (!this.OutputsActivated()) {
                foreach (var neuron in Neurons) {
                    if(neuron.Value.Type != NeuronType.Input && neuron.Value.Type != NeuronType.Bias) neuron.Value.ResetState();

                    foreach (int connection in neuron.Value.IncommingConnections) {
                        Neuron src = Neurons[Connections[connection].SourceID];
                        if (src.Activated || src.Type == NeuronType.Bias || src.Type == NeuronType.Input) src.Activate();

                        src.Input(Connections[connection].Weight * src.Value);
                    }
                }

                foreach (var neuron in Neurons) {
                    if (neuron.Value.Type != NeuronType.Input && neuron.Value.Type != NeuronType.Bias) {
                        if (neuron.Value.Activated) {
                            neuron.Value.Activate();
                        }
                    }
                }
            }
        }

        //adds the inverse lerped value to fitness
        public void AddFitness(float value, float minValue, float maxValue) {
            Fitness += NNUtility.InverseLerp(minValue, maxValue, value);
        }

        //adds value directly to fitness
        public void AddFitness(float value) {
            Fitness += value;
        }

        public void ResetFitness() {
            Fitness = 0f;
        }

    }
}
