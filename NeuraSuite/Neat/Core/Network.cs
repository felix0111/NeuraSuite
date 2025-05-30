using System;
using System.Collections.Generic;
namespace NeuraSuite.Neat.Core {
    public class Network {

        //the underlying genome
        public Genome Genome;

        //dict of every NodeGene, identified by its id
        private Dictionary<int, NodeGene> _nodes = new();

        //dict of every node value, identified by its id
        private Dictionary<int, double> _nodeValues = new();

        //sorted by the endId which should make evaluation more efficient
        private Dictionary<int, List<ConnectionGene>> _connections = new();

        /// <summary>
        /// Create a new network from a genome. Modifying the genome after creation does not reflect to the network!
        /// </summary>
        public Network(Genome genome) {
            Genome = genome;

            //pre-sort connections by endId
            foreach (var connection in Genome.Connections.Values) {
                //no need for disabled connections
                if(!connection.Enabled) continue;

                _connections.TryAdd(connection.EndId, new List<ConnectionGene>());
                _connections[connection.EndId].Add(connection);
            }

            foreach (var node in Genome.Nodes.Values) {
                //if node not input and does not have incomming connections, dont add
                if(node.Type != NodeType.Input && !_connections.ContainsKey(node.Id)) continue;

                _nodes.Add(node.Id, node);
                _nodeValues.Add(node.Id, 0);
            }
        }

        /// <summary>
        /// Set the value of a node. Values of input nodes only have to be set once.
        /// </summary>
        public void SetValue(int nodeId, double value) {
            //if node is not present, skip
            if (!_nodeValues.ContainsKey(nodeId)) return;
            _nodeValues[nodeId] = value;
        }

        /// <summary>
        /// Get the value of a node.
        /// </summary>
        public double GetValue(int nodeId) => _nodeValues.TryGetValue(nodeId, out var value) ? value : 0D;

        /// <summary>
        /// Feed-Forward all node-values and activate all nodes (except input nodes).
        /// </summary>
        /// <param name="passes">The amount of times the Feed-Forward process is done. Used to reduce signal delay in larger networks.</param>
        public void Evaluate(int passes) {
            for (int i = 0; i < passes; i++) {
                foreach (var node in _nodes) {
                    //skip input nodes
                    if(node.Value.Type == NodeType.Input) continue;

                    //foreach connection that ends at this neuron
                    double sum = 0;
                    foreach (var connection in _connections[node.Key]) {
                        sum += GetValue(connection.StartId) * connection.Weight;
                    }

                    //apply sigmoid and update value
                    SetValue(node.Key, 1D / (1D + Math.Exp(-4.9D * sum)));
                }
            }
        }

        /// <summary>
        /// Resets all node values to 0D.
        /// </summary>
        public void Reset() {
            foreach (var nodeValue in _nodeValues) {
                _nodeValues[nodeValue.Key] = 0D;
            }
        }
    }
}
