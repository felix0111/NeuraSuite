using System;
using System.Linq;
using NeuraSuite.Neat.Core;

namespace NeuraSuite.Neat.Utility
{
    public static class Mutator {

        private static Random _random = new ();

        /// <summary>
        /// Randomly mutates a genome according to the specified MutationOptions.
        /// </summary>
        public static void Mutate(this Genome g, InnovationManager im, MutationSettings settings) {
            //chance to add connection
            if (_random.NextDouble() <= settings.AddConnectionChance) {
                var validStartNodes = g.Nodes.Values.Where(o => o.Type != NodeType.Output).ToArray();
                var validEndNodes = g.Nodes.Values.Where(o => o.Type != NodeType.Input).ToArray();

                //retry 4 times at max
                for (int i = 0; i < 4; i++) {
                    var start = validStartNodes[_random.Next(validStartNodes.Length)];
                    var end = validEndNodes[_random.Next(validEndNodes.Length)];

                    //if the add connection did not fail, break out of loop
                    if (g.AddConnection(im.GetInnovation(start.Id, end.Id), start.Id, end.Id, _random.RandomWeight())) break;
                }
            }

            //chance to split connection
            if (_random.NextDouble() <= settings.SplitConnectionChance && g.Connections.Count > 0) {
                //only enabled connections
                var validConnections = g.Connections.Values.Where(o => o.Enabled).ToArray();
                
                //mutation not possible if no enabled connections
                if (validConnections.Length > 0) {
                    var con = validConnections[_random.Next(validConnections.Length)];
                    int newNodeId = im.NewNodeId;

                    g.SplitConnection(con.Innovation, newNodeId, im.GetInnovation(con.StartId, newNodeId), im.GetInnovation(newNodeId, con.EndId));
                }
            }

            //every gene has a chance to change weight
            foreach (var connection in g.Connections.ToList()) {
                if(_random.NextDouble() > settings.ChangeWeightChance) continue;

                //10% chance to reset weight, 90% chance to change weight by MaxWeightDelta
                double newWeight;
                if (_random.NextDouble() < 0.1D) {
                    newWeight = _random.RandomWeight();
                } else {
                    newWeight = connection.Value.Weight + _random.RandomWeight(settings.MaxWeightDelta);
                }

                g.ChangeWeight(connection.Value.Innovation, newWeight);
            }
        }
    }
}
