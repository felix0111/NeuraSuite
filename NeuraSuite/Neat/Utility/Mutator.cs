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
        public static void Mutate(this Genome g, InnovationManager im, MutationOptions options) {
            //chance to add connection
            if (_random.NextDouble() <= options.AddConnectionChance) {
                var neurons = g.Nodes.Keys.ToArray();

                var start = neurons[_random.Next(neurons.Length)];
                var end = neurons[_random.Next(neurons.Length)];
                g.AddConnection(im.GetInnovation(start, end), start, end);
            }

            //chance to split connection
            if (_random.NextDouble() <= options.SplitConnectionChance) {
                if (g.Connections.Count == 0) return;

                var con = g.Connections.Values.ToArray()[_random.Next(g.Connections.Count)];
                if (!con.Enabled) return;

                int newNodeId = im.NewNodeId;
                g.SplitConnection(con.Innovation, newNodeId, im.GetInnovation(con.StartId, newNodeId), im.GetInnovation(newNodeId, con.EndId));
            }

            //chance to change weight (or reset to new random value)
            if (_random.NextDouble() <= options.ChangeWeightChance) {
                if (g.Connections.Count == 0) return;

                var con = g.Connections.Values.ToArray()[_random.Next(g.Connections.Count)];

                //10% chance to reset weight, 90% chance to change weight by MaxWeightDelta
                if (_random.NextDouble() < 0.1D) {
                    g.ChangeWeight(con.Innovation, (_random.Next(0, 2) * 2D - 1D) * _random.NextDouble());
                } else {
                    g.ChangeWeight(con.Innovation, con.Weight + (_random.Next(0, 2) * 2D - 1D) * _random.NextDouble() * options.MaxWeightDelta);
                }
            }
        }
    }

    public struct MutationOptions {

        public double AddConnectionChance, SplitConnectionChance, ChangeWeightChance, MaxWeightDelta;

        public MutationOptions(double addConnectionChance, double splitConnectionChance, double changeWeightChance, double maxWeightDelta) {
            AddConnectionChance = addConnectionChance;
            SplitConnectionChance = splitConnectionChance;
            ChangeWeightChance = changeWeightChance;
            MaxWeightDelta = maxWeightDelta;
        }

    }
}
