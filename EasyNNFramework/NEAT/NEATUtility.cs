using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyNNFramework.NEAT {
    public static class NEATUtility {

        public static bool CheckRecurrent(this Network network, int sourceID, int targetID) {

            if (network.GetNeuronType(sourceID) == NeuronType.Action) return true;  //connection starting at action neuron
            if (network.GetNeuronType(sourceID) == NeuronType.Input || network.GetNeuronType(sourceID) == NeuronType.Bias) return false;  //connection starting at input neuron
            if (network.GetNeuronType(targetID) == NeuronType.Action) return false; //connection ending at action neuron

            //check if target neuron exists in incomming connections
            foreach (int connectionID in network.Neurons[sourceID].IncommingConnections) {
                Connection con = network.Connections[connectionID];
                if (con.SourceID == targetID) return true;

                //search for target in incomming connections of current source neuron
                if (network.CheckRecurrent(con.SourceID, targetID)) return true;
            }

            return false;

        }

        public static NeuronType GetNeuronType(this Network network, int neuronID) {
            return network.Neurons[neuronID].Type;
        }

        public static Connection GetConnection(this Network network, int connectionID) {
            if (network.Connections.TryGetValue(connectionID, out Connection c1)) return c1;
            if (network.RecurrentConnections.TryGetValue(connectionID, out Connection c2)) return c1;
            throw new Exception("Could not find connection with innovation ID " + connectionID);
        }

        public static Connection RandomConnectionType(this Network network, Random rnd, bool recurrent) {
            return recurrent ? network.RecurrentConnections.ToList()[rnd.Next(0, network.RecurrentConnections.Count)].Value : network.Connections.ToList()[rnd.Next(0, network.Connections.Count)].Value;
        }

        public static Connection RandomConnection(this Network network, Random rnd) {
            var allConns = network.Connections.Concat(network.RecurrentConnections).ToList();
            return allConns[rnd.Next(0, allConns.Count)].Value;
        }

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

        public static bool ExistsConnection(this Network network, int innovID) {
            return network.Connections.ContainsKey(innovID) || network.RecurrentConnections.ContainsKey(innovID);
        }

        public static List<Neuron> GetUselessHidden(this Network network) {
            List<Neuron> useless = new List<Neuron>();

            foreach (Neuron n in network.HiddenNeurons) {

                if (n.IncommingConnections.Count == 0 || n.OutgoingConnections.Count == 0) {
                    useless.Add(n);
                }
            }

            return useless;
        }

        public static bool OutputsActivated(this Network network) {
            for (int i = 0; i < network.ActionNeurons.Length; i++) {
                if (!network.ActionNeurons[i].Activated) return false;
            }

            return true;
        }

        //returns tuple of Item1=matching, Item2=disjoint, Item3=excess
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

        public static float Distance(in Network network1, in Network network2, SpeciationOptions options) {
            var split = GetMatchingAndDisjoint(network1, network2);

            int N = Math.Max(network1.Connections.Count + network1.RecurrentConnections.Count, network2.Connections.Count + network2.RecurrentConnections.Count);
            if (N == 0) return 0f;

            //calculates the weight distance of all matching connections
            //for example: w1 = 0.5 && 0.1; w2 = -1 & 1 => delta = 2.4
            float W = 0f;
            for (int i = 0; i < split.Item1.Length; i++) {
                float w1 = network1.GetConnection(split.Item1[i]).Weight;
                float w2 = network2.GetConnection(split.Item1[i]).Weight;

                W += Math.Abs(w1 - w2);
            }

            float delta = (options.DisjointFactor * split.Item2.Length) / N;
            delta += options.WeightFactor * W;

            return delta;
        }
    }
}
