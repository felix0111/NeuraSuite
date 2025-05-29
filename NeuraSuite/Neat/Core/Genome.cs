using System.Collections.Generic;

namespace NeuraSuite.Neat.Core {
    public class Genome {

        public Dictionary<int, NodeGene> Nodes = new();
        public Dictionary<int, ConnectionGene> Connections = new();

        public Genome(List<NodeGene> defaultNodes = null, List<ConnectionGene> defaultConnections = null) {
            if (defaultNodes != null) {
                foreach (var node in defaultNodes) Nodes.Add(node.Id, node);
            }

            if (defaultConnections != null) {
                foreach (var connection in defaultConnections) Connections.Add(connection.Innovation, connection);
            }
        }

        public bool AddConnection(int innovation, int startId, int endId) => Connections.TryAdd(innovation, new ConnectionGene(innovation, startId, endId));

        public bool ToggleConnection(int innovation) {
            if (!Connections.ContainsKey(innovation)) return false;

            var con = Connections[innovation];
            con.Enabled = !con.Enabled;

            Connections[innovation] = con;
            return true;
        }

        public bool ChangeWeight(int innovation, double weight) {
            if (!Connections.ContainsKey(innovation)) return false;

            var con = Connections[innovation];
            con.Weight = weight;

            Connections.Remove(innovation);
            Connections.Add(innovation, con);
            return true;
        }

        public bool SplitConnection(int oldInnovation, int newNodeId, int innovationStartToNew, int innovationNewToEnd) {
            if (!Connections.ContainsKey(oldInnovation) || Nodes.ContainsKey(newNodeId)) return false;

            var old = Connections[oldInnovation];

            //disable old connection and add two new connections
            if (old.Enabled) ToggleConnection(oldInnovation);
            Connections.Add(innovationStartToNew, new ConnectionGene(innovationStartToNew, old.StartId, newNodeId));
            Connections.Add(innovationNewToEnd, new ConnectionGene(innovationNewToEnd, newNodeId, old.EndId));

            //add the new node
            Nodes.Add(newNodeId, new NodeGene(newNodeId, NodeType.Hidden));

            return true;
        }

        public Genome Clone() {
            Genome g = new();

            foreach (var node in Nodes) g.Nodes.Add(node.Key, node.Value);
            foreach (var connection in Connections) g.Connections.Add(connection.Key, connection.Value);

            return g;
        }
    }

    public struct NodeGene(int id, NodeType type) {
        public int Id = id;
        public NodeType Type = type;
    }

    public struct ConnectionGene(int innovation, int startId, int endId, double weight = 1, bool enabled = true) {
        public int Innovation = innovation;
        public int StartId = startId;
        public int EndId = endId;
        public double Weight = weight;
        public bool Enabled = enabled;
    }

    public enum NodeType {
        Input,
        Hidden,
        Output
    }
}