using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuraSuite.Neat.Core {
    public class Species {

        public Genome Representative;
        public List<Genome> Members = new();

        public Species(Genome representative) {
            Representative = representative.Clone();
        }

        /// <summary>
        /// Gets a random member. Genomes with higher fitness have a higher chance.
        /// </summary>
        public Genome RandomByFitness(Random r) {
            if(Members.Count == 0) return null;

            var fitnessSum = Members.Sum(o => o.Fitness);
            var normalizedFitnesses = Members.Select(o => (o, o.Fitness / fitnessSum)).OrderBy(o => o.Item2).ToList();

            double rnd = r.NextDouble();
            double probabilitySum = 0D;
            foreach (var genome in normalizedFitnesses) {
                probabilitySum += genome.Item2;
                if (rnd <= probabilitySum) return genome.o;
            }

            throw new Exception("Should not happen!");
        }

        /// <summary>
        /// Removes the specified amount of low-performing genomes from the species.
        /// </summary>
        /// <param name="percentage">Must be less or equal to 1D.</param>
        public void RemoveWorstMembers(double percentage) {
            if(percentage <= 0D) return;

            int amount = (int)Math.Round(Members.Count * percentage);

            Members = Members.OrderBy(o => o.Fitness).Skip(amount).ToList();
        }

        /// <summary>
        /// Checks if a genome is compatible to the species.
        /// </summary>
        public bool Compatible(Genome genome, double threshold) {
            return Distance(genome, Representative, 1D, 1D, 0.4D) < threshold;
        }

        /// <summary>
        /// Calculates the distance of two genomes by comparing their matching, disjoint and excess genes including difference in weights.
        /// Disabled genes are also taken into account. 
        /// </summary>
        public static double Distance(Genome genome1, Genome genome2, double excessFactor, double disjointFactor, double weightFactor) {
            var (matching, disjoint, excess) = MatchingDisjointExcess(genome1, genome2);

            //number of genes in the larger genome
            double n = Math.Max(genome1.Connections.Count, genome2.Connections.Count);
            if (n < 20) n = 1;

            //calculates the average weight difference of all matching genes
            double w = 0D;
            foreach (var innovation in matching) {
                w += Math.Abs(genome1.Connections[innovation].Weight - genome2.Connections[innovation].Weight);
            }
            w /= matching.Length;

            return (excessFactor*excess.Length)/n + (disjointFactor*disjoint.Length)/n + weightFactor * w;
        }

        /// <summary>
        /// Compares two genomes and returns all matching, disjoint and excess genes.
        /// </summary>
        /// <returns>Returns Tuple of type (matching genes, disjoint genes, excess genes).</returns>
        public static Tuple<int[], int[], int[]> MatchingDisjointExcess(Genome genome1, Genome genome2) {
            List<int> matching = new List<int>();
            List<int> disjoint = new List<int>();
            List<int> excess = new List<int>();

            int lowestInnovation = Math.Min(genome1.Connections.Keys.Max(), genome2.Connections.Keys.Max());

            foreach (var connection in genome1.Connections) {
                if (genome2.Connections.ContainsKey(connection.Key)) {
                    matching.Add(connection.Key);
                } else {
                    if (connection.Key > lowestInnovation) {
                        excess.Add(connection.Key);
                    } else {
                        disjoint.Add(connection.Key);
                    }
                }
            }

            foreach (var connection in genome2.Connections) {
                if (genome1.Connections.ContainsKey(connection.Key)) {
                    //matching connections has already been added
                } else {
                    if (connection.Key > lowestInnovation) {
                        excess.Add(connection.Key);
                    } else {
                        disjoint.Add(connection.Key);
                    }
                }
            }

            return new Tuple<int[], int[], int[]>(matching.ToArray(), disjoint.ToArray(), excess.ToArray());
        }
    }
}
