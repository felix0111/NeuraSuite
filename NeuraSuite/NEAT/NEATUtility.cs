using System;
using System.Collections.Generic;
using System.Linq;
using Random = System.Random;

namespace NeuraSuite.Neat {
    public static class NEATUtility {

        /// <summary>
        /// Checks if a connection between two neurons would be recurrent.
        /// </summary>
        // TODO might need to do more testing
        public static bool CheckRecurrent(this Network network, int sourceID, int targetID) {

            if (sourceID == targetID) return true;  //connection going to self => recurrent

            if (network.GetNeuronType(sourceID) == NeuronType.Input || network.GetNeuronType(sourceID) == NeuronType.Bias) return false;  //connection starting at input neuron => not recurrent
            if (network.GetNeuronType(targetID) == NeuronType.Action) return false; //connection ending at action neuron => not recurrent

            //check if target neuron exists in incomming connections
            foreach (int innovID in network.Neurons[sourceID].IncommingConnections) {
                Connection con = network.Connections[innovID];
                if (con.SourceID == targetID) return true;

                //search for target in incomming connections of current source neuron
                if (network.CheckRecurrent(con.SourceID, targetID)) return true;
            }

            return false;

        }

        /// <summary>
        /// Returns the type of a specific neuron in a network.
        /// </summary>
        public static NeuronType GetNeuronType(this Network network, int neuronID) {
            return network.Neurons[neuronID].Type;
        }

        /// <summary>
        /// Get a specific connection of a network by innovation ID.
        /// </summary>
        /// <exception cref="Exception">A connection with the specified ID could not be found.</exception>
        public static Connection GetConnection(this Network network, int innovationID) {
            if (network.Connections.TryGetValue(innovationID, out Connection c1)) return c1;
            if (network.RecurrentConnections.TryGetValue(innovationID, out Connection c2)) return c1;
            throw new Exception("Could not find connection with innovation ID: " + innovationID);
        }

        /// <summary>
        /// Returns a random recurrent or normal connection of a network.
        /// </summary>
        /// <param name="recurrent">If only recurrent connections should be chosen.</param>
        public static Connection RandomConnectionType(this Network network, Random rnd, bool recurrent) {
            return recurrent ? network.RecurrentConnections.ToList()[rnd.Next(0, network.RecurrentConnections.Count)].Value : network.Connections.ToList()[rnd.Next(0, network.Connections.Count)].Value;
        }

        /// <summary>
        /// Returns a random (recurrent) connection of a network.
        /// </summary>
        public static Connection RandomConnection(this Network network, Random rnd) {
            var allConns = network.Connections.Concat(network.RecurrentConnections).ToList();
            return allConns[rnd.Next(0, allConns.Count)].Value;
        }

        /// <summary>
        /// Checks if a connection exists in a network by using a source and target neuron.
        /// </summary>
        public static bool ExistsConnection(this Network network, int sourceID, int targetID) {
            
            Connection buffer;
            for (int i = 0; i < network.RecurrentConnections.Count; i++) {
                buffer = network.RecurrentConnections.Values.ElementAt(i);
                if (buffer.SourceID == sourceID && buffer.TargetID == targetID) return true;
            }

            for (int i = 0; i < network.Connections.Count; i++) {
                buffer = network.Connections.Values.ElementAt(i);
                if (buffer.SourceID == sourceID && buffer.TargetID == targetID) return true;
            }

            return false;
        }

        /// <summary>
        /// Checks if a connection exists in a network.
        /// </summary>
        /// <param name="innovID">Unique identifier for conections in a population.</param>
        public static bool ExistsConnection(this Network network, int innovID) {
            return network.Connections.ContainsKey(innovID) || network.RecurrentConnections.ContainsKey(innovID);
        }

        /// <summary>
        /// Checks if a neuron exists in a network.
        /// </summary>
        public static bool ExistsNeuron(this Network network, int neuronID) {
            return network.Neurons.ContainsKey(neuronID);
        }

        /// <summary>
        /// Finds all useless hidden neurons a network may have.
        /// </summary>
        public static List<Neuron> GetUselessHidden(this Network network) {
            List<Neuron> useless = new List<Neuron>();

            foreach (Neuron n in network.HiddenNeurons) {
                if (n.IncommingConnections.Count == 0 || n.OutgoingConnections.Count == 0) {
                    useless.Add(n);
                }
            }

            return useless;
        }

        /// <summary>
        /// Used to check if all output neurons of a network are activated.
        /// </summary>
        public static bool OutputsActivated(this Network network) {
            for (int i = 0; i < network.ActionNeurons.Length; i++) {
                if (!network.ActionNeurons[i].Activated) return false;
            }

            return true;
        }

        /// <summary>
        /// Used to get the matching and disjoint connections of 2 networks.
        /// <br/> <br/>
        /// Returns tuple: (array of matching, array of disjoint)
        /// </summary>
        public static (int[], int[]) GetMatchingAndDisjoint(in Network network1, in Network network2) {
            List<int> disjoint = new List<int>();
            List<int> matching = new List<int>();

            foreach (var connection in network1.Connections.Concat(network1.RecurrentConnections)) {
                if (network2.ExistsConnection(connection.Key)) {
                    matching.Add(connection.Key);
                } else {
                    disjoint.Add(connection.Key);
                }
            }

            foreach (var connection in network2.Connections.Concat(network2.RecurrentConnections)) {
                if (network1.ExistsConnection(connection.Key)) {
                    //matching connections have already been added
                } else {
                    disjoint.Add(connection.Key);
                }
            }

            //if anything left in collected, excess genes!
            return (matching.ToArray(), disjoint.ToArray());
        }

        /// <summary>
        /// This function evaluates the difference of two networks. <br/>
        /// Distance = 0 no difference <br/> Distance > 0 some differences <br/> Distance >>> 0 many differences
        /// <br/> <br/>
        /// Returns <code>disjoint-delta + weight-delta</code>
        /// <br/>
        /// Disjoint Delta: (<see cref="SpeciationOptions.DisjointFactor"/> * amount of disjoints) / number of connections
        /// <br/>
        /// Weight Delta: sum( abs(weight1 - weight2) ) * <see cref="SpeciationOptions.WeightFactor"/>
        /// </summary>
        public static float Distance(in Network network1, in Network network2, SpeciationOptions options) {
            //tuple structure: (matching, disjoint)
            var split = GetMatchingAndDisjoint(network1, network2);

            //get highest amount of connections
            int N = Math.Max(network1.Connections.Count + network1.RecurrentConnections.Count, network2.Connections.Count + network2.RecurrentConnections.Count);
            //if both have no connections, return 0
            if (N == 0) return 0f;

            //calculates the absolute difference of the weights of all matching connections
            //higher delta, higher difference
            //for example: weight_1 = 0.5 && 0.1; w2 = -1 & 1 => delta = 2.4
            float W = 0f;
            for (int i = 0; i < split.Item1.Length; i++) {
                float w1 = network1.GetConnection(split.Item1[i]).Weight;
                float w2 = network2.GetConnection(split.Item1[i]).Weight;

                W += Math.Abs(w1 - w2);
            }

            //more disjoints = higher delta, bigger difference
            //for example: disjointfactor = 1; disjoints = 10; number of weights = 20; delta = 0.5
            //for example: disjointfactor = 1; disjoints = 5; number of weights = 5; delta = 1
            float delta = (options.DisjointFactor * split.Item2.Length) / N;
            delta += options.WeightFactor * W;

            return delta;
        }
    }
}
