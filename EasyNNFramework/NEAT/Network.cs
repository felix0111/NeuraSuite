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

        //TODO set protected or private to discourage direct manipulation of neurons? 
        public Dictionary<int, Neuron> Neurons = new Dictionary<int, Neuron>();

        public Dictionary<int, Connection> Connections = new Dictionary<int, Connection>();
        public Dictionary<int, Connection> RecurrentConnections = new Dictionary<int, Connection>();

        /// <summary>
        /// This buffer is used to feed values to the network.
        /// </summary>
        public float[] InputValues;

        /// <summary>
        /// This buffer is used to store the output of the network.
        /// </summary>
        public float[] OutputValues;

        /// <summary>
        /// When creating neurons, this will be used to determine their new ID.
        /// </summary>
        //TODO might be inefficient with very big amount of neurons
        public int GetFreeNeuronID => Neurons.Keys.Max() + 1;

        /// <summary>
        /// Used to differentiate between multiple networks.
        /// </summary>
        public readonly int NetworkID;

        public int SpeciesID;
        public float Fitness;

        private int Depth => _ls.LayerArray.Count;

        /// <summary>
        /// Used for calculating the network. This determines the order in which the neurons get activated in the feed forward process.
        /// </summary>
        private LayerStructure _ls;

        /// <summary>
        /// Determines if useless/unconnected hidden neurons get automatically removed.
        /// </summary>
        public bool AllowUselessHidden {
            get => _allowUselessHidden;
            set {
                _allowUselessHidden = value;
                RecalculateStructure(!_allowUselessHidden);
            }
        }
        private bool _allowUselessHidden;

        /// <summary>
        /// Creates a new network. Use correct neuron ID's!
        /// </summary>
        /// <exception cref="Exception">Multiple neurons with the same ID.</exception>
        public Network(int networkID, Neuron[] inputTemplate, Neuron[] outputTemplate) {

            NetworkID = networkID;

            try {
                foreach (var neuron in inputTemplate) {
                    Neurons.Add(neuron.ID, new Neuron(neuron.ID, neuron.Function, NeuronType.Input));
                }

                foreach (var neuron in outputTemplate) {
                    Neurons.Add(neuron.ID, new Neuron(neuron.ID, neuron.Function, NeuronType.Action));
                }
            } catch (ArgumentException _) {
                throw new Exception("Make sure to give every template neuron a unique ID!");
            }

            RecalculateNeuronArrays();
            RecalculateStructure(!AllowUselessHidden);

            InputValues = new float[InputNeurons.Length];
            OutputValues = new float[ActionNeurons.Length];
        }

        /// <summary>
        /// Will create a new network based of another one. Will also copy species identifier and fitness!
        /// </summary>
        //TODO update InnovationCollection in Neat class?
        public Network(int networkID, in Network templateNetwork) {

            NetworkID = networkID;
            _allowUselessHidden = templateNetwork.AllowUselessHidden;

            foreach (var neuron in templateNetwork.Neurons.Values) {
                Neurons.Add(neuron.ID, neuron.Clone());
            }

            foreach (var connection in templateNetwork.Connections.Values) {
                Connections.Add(connection.InnovationID, connection);
            }

            foreach (var connection in templateNetwork.RecurrentConnections.Values) {
                RecurrentConnections.Add(connection.InnovationID, connection);
            }

            RecalculateNeuronArrays();
            RecalculateStructure(!AllowUselessHidden);

            SpeciesID = templateNetwork.SpeciesID;
            Fitness = templateNetwork.Fitness;

            InputValues = new float[InputNeurons.Length];
            OutputValues = new float[ActionNeurons.Length];
        }


        /// <summary>
        /// Will sort all neurons by Input-, Hidden- and Output.
        /// </summary>
        // TODO Inefficient because LINQ but it's only used when adding/removing neurons so whatever I guess
        private void RecalculateNeuronArrays() {
            InputNeurons = Neurons.Values.Where(o => o.Type == NeuronType.Input || o.Type == NeuronType.Bias).ToArray();
            HiddenNeurons = Neurons.Values.Where(o => o.Type == NeuronType.Hidden).ToArray();
            ActionNeurons = Neurons.Values.Where(o => o.Type == NeuronType.Action).ToArray();
        }

        /// <summary>
        /// Will reevaluate the structure of the neural network to make it more efficient for computing.
        /// This is normally done automatically after adding/removing neurons/connections.
        /// <br/> <br/>
        /// Useless/Unconnected hidden neurons will be removed!
        /// </summary>
        //TODO possible infinite loop?
        public void RecalculateStructure(bool removeUselessHidden) {

            if (removeUselessHidden) {
                bool temp = AllowUselessHidden;
                _allowUselessHidden = true;

                List<Neuron> useless = this.GetUselessHidden();
                do {
                    foreach (var n in useless) {
                        RemoveNeuron(n.ID);
                    }

                    useless = this.GetUselessHidden();
                } while (useless.Count != 0);

                _allowUselessHidden = temp;
            }

            //create a new layer structure
            _ls = new LayerStructure(this);
        }
        
        /// <summary>
        /// Replaces a connection with a neuron.
        /// <br/>
        /// The new connection leading to the new neuron will be the same weight as the old. The second connection will get a weight of 1.
        /// <br/> <br/>
        /// innovID: the unique ID associated to the connection
        /// </summary>
        /// <exception cref="Exception">Can't add a neuron between a recurrent connection.</exception>
        /// <exception cref="Exception">The connection with the specified ID could not be found.</exception>
        public Neuron AddNeuron(Neat neat, int innovID, ActivationFunction function) {
            // check if connection exists
            if (RecurrentConnections.ContainsKey(innovID)) throw new Exception("Can't add neuron between recurent connection!");
            if (!Connections.ContainsKey(innovID)) throw new Exception($"Connection with ID {innovID} does not exist!");

            Connection con = Connections[innovID];
            Neuron newNeuron = new Neuron(GetFreeNeuronID, function, NeuronType.Hidden);
            Neurons.Add(newNeuron.ID, newNeuron);

            RecalculateNeuronArrays();

            bool temp = _allowUselessHidden;
            _allowUselessHidden = true;

            RemoveConnection(con.InnovationID);
            AddConnection(neat.NewInnovation(con.SourceID, newNeuron.ID), con.SourceID, newNeuron.ID, con.Weight);
            AddConnection(neat.NewInnovation(newNeuron.ID, con.TargetID), newNeuron.ID, con.TargetID, 1f);

            _allowUselessHidden = temp;

            RecalculateStructure(!AllowUselessHidden);

            return newNeuron;
        }

        /// <summary>
        /// Adds a neuron to the network without any connections.
        /// <br/>
        /// Set <see cref="AllowUselessHidden"/> true or else the neuron would be removed immediately!
        /// </summary>
        /// <exception cref="Exception"><see cref="AllowUselessHidden"/> is set to false!</exception>
        public Neuron AddNeuronUnsafe(Neuron n) {
            if (!AllowUselessHidden) throw new Exception("You must set AllowUselessHidden true!");

            if (Neurons.ContainsKey(n.ID)) return null;

            Neuron newNeuron = n.Clone();
            Neurons.Add(newNeuron.ID, newNeuron);

            RecalculateNeuronArrays();
            RecalculateStructure(!AllowUselessHidden);
            
            return newNeuron;
        }

        /// <summary>
        /// Removes a neuron from the network. Will also remove all connections going to and from this neuron which may also lead to the removal of other neurons!
        /// <br/>
        /// If you want to keep neurons which may get unconnected, set <see cref="AllowUselessHidden"/> true!
        /// </summary>
        /// <exception cref="Exception">Neuron with the specified ID does not exists.</exception>
        public void RemoveNeuron(int neuronID) {
            if (!Neurons.ContainsKey(neuronID)) throw new Exception("Neuron does not exist!");

            bool temp = _allowUselessHidden;
            _allowUselessHidden = true;

            //remove all connections that are connected to this neuron id
            RemoveDependingConnections(neuronID);

            //remove actual neuron from array
            Neurons.Remove(neuronID);

            _allowUselessHidden = temp;

            RecalculateNeuronArrays();
            RecalculateStructure(!AllowUselessHidden);
        }
        
        /// <summary>
        /// Adds a connection between two neurons. If connection already exists, returns connection struct with all values being -1.
        /// <br/> <br/>
        /// Get the innovation ID by using <see cref="Neat.NewInnovation(int, int)"/>. This is a unique identifier for all connections.
        /// <br/>
        /// If you're not managing networks with the NEAT class, use a unique number for each connection.
        /// </summary>
        /// <returns>The newly created connection. NOT A REFERENCE!</returns>
        // TODO improve handling when a connection already exists, maybe exception? or use "out bool success"?
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

            RecalculateStructure(!AllowUselessHidden);

            return newConnection;
        }

        /// <summary>
        /// Changes the weight of a (recurrent) connection.
        /// </summary>
        /// <exception cref="Exception">Connection with specified ID not found.</exception>
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

        /// <summary>
        /// (De)Activates a connection.
        /// <br/> <br/>
        /// This is used when a connection is not needed at the moment but it should still be part of the network and thus part of the evolutionary process.
        /// </summary>
        /// <exception cref="Exception">Connection with specified ID not found.</exception>
        // TODO ToggleConnection -> SetConnectionState, might be a better name for this function
        public void ToggleConnection(int connectionID, bool active) {
            if (RecurrentConnections.TryGetValue(connectionID, out Connection val)) {
                val.Activated = active;
                RecurrentConnections[connectionID] = val;
            } else if (Connections.TryGetValue(connectionID, out Connection val2)) {
                val2.Activated = active;
                Connections[connectionID] = val2;
            } else {
                throw new Exception("Connection with innovation ID " + connectionID + " was not found!");
            }
        }

        /// <summary>
        /// Removes a connection with the specified ID. Some previously connected neurons may get removed!
        /// </summary>
        /// <exception cref="Exception">Connection with specified ID not found.</exception>
        public void RemoveConnection(int connectionID) {
            //if connection found
            if (Connections.TryGetValue(connectionID, out Connection val)) {
                Connections.Remove(connectionID);
                Neurons[val.SourceID].OutgoingConnections.Remove(connectionID);
                Neurons[val.TargetID].IncommingConnections.Remove(connectionID);
            } else if(!RecurrentConnections.Remove(connectionID)) {
                throw new Exception("Connection with innovation ID " + connectionID + " was not found!");
            }

            RecalculateStructure(!AllowUselessHidden);
        }

        /// <summary>
        /// Removes all connections which are associated with a specific neuron.
        /// <br/>
        /// Set <see cref="AllowUselessHidden"/> true, if you want to keep the neuron!
        /// </summary>
        public void RemoveDependingConnections(int neuronID) {
            foreach (var connection in Connections.ToList()) {
                if (connection.Value.SourceID == neuronID || connection.Value.TargetID == neuronID) RemoveConnection(connection.Value.InnovationID);
            }

            foreach (var connection in RecurrentConnections.ToList()) {
                if (connection.Value.SourceID == neuronID || connection.Value.TargetID == neuronID) RemoveConnection(connection.Value.InnovationID);
            }
        }


        /// <summary>
        /// Resets the state of all neurons (including their internal states which are used for recurrent connections!)
        /// </summary>
        // ResetState is called two times so that the internal buffer (LastValue) clears
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

            Fitness = 0f;
        }


        /// <summary>
        /// Feeds network with the values in "InputValues" buffer and executes the neural network. The result will be stored in the "OutputValues" buffer.
        /// </summary>
        /// <exception cref="Exception">A neuron was not activated in the feed forward process.</exception>
        // TODO apply recurrent connections before resetting all neurons? or at the end of the function?
        // TODO check if all neurons actually execute only one time?!
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
            for (int layerIndex = 0; layerIndex < _ls.LayerArray.Count; layerIndex++) {
                //for each neuron id in layer
                for (int i = 0; i < _ls.LayerArray[layerIndex].Count; i++) {
                    Neuron target = Neurons[_ls.LayerArray[layerIndex][i]];

                    for (int j = 0; j < target.IncommingConnections.Count; j++) {
                        Connection con = Connections[target.IncommingConnections[j]];
                        Neuron src = Neurons[con.SourceID];

                        //shouldnt happen
                        if (!src.Activated) throw new Exception("The neuron " + src.ID + " was not activated in the feed forward process!");
                        
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

        /// <summary>
        /// Normally the network gets calculated in a feed forward style.
        /// This function partially calculates the neurons in an unsorted order which reduces computing time.
        /// <br/> <br/>
        /// Does not account for recurrent connections at the moment.
        /// </summary>
        // TODO function still work in progress
        public void CalculateStep() {

            //feed and activate each neuron values
            for (int i = 0; i < InputNeurons.Length; i++) {
                InputNeurons[i].ResetState();
                InputNeurons[i].Input(InputValues[i]);
                InputNeurons[i].Activate();
            }

            //feed and activate each hidden neuron
            foreach (Neuron hiddenNeuron in HiddenNeurons) {
                hiddenNeuron.ResetState();
                foreach (int connection in hiddenNeuron.IncommingConnections) {
                    Neuron src = Neurons[Connections[connection].SourceID];
                    hiddenNeuron.Input(Connections[connection].Weight * src.Value);
                }
                hiddenNeuron.Activate();
            }



            //feed and activate each output neuron
            for (int i = 0; i < ActionNeurons.Length; i++) {
                ActionNeurons[i].ResetState();
                foreach (int connection in ActionNeurons[i].IncommingConnections) {
                    Neuron src = Neurons[Connections[connection].SourceID];
                    ActionNeurons[i].Input(Connections[connection].Weight * src.Value);
                }

                ActionNeurons[i].Activate();
                OutputValues[i] = ActionNeurons[i].Value;
            }
        }

    }
}
