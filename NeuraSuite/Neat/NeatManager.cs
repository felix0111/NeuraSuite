using NeuraSuite.Neat.Core;
using NeuraSuite.Neat.Utility;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuraSuite.Neat {
    public class NeatManager {

        public InnovationManager InnovationManager;

        public MutationOptions MutationOptions;
        public NeatOptions NeatOptions;

        public List<Genome> EntirePopulation = new();
        public List<Species> Species = new();

        private List<Network> _phenotypes = new();

        private Random _random = new ();

        public NeatManager(Genome initialGenome, NeatOptions neatOptions, MutationOptions mutationOptions) {
            //adjusts innovation and node counter to initial genome
            InnovationManager = new InnovationManager(initialGenome);

            NeatOptions = neatOptions;
            MutationOptions = mutationOptions;

            //create start population
            for (int i = 0; i < NeatOptions.TargetPopulationSize; i++) EntirePopulation.Add(initialGenome.Clone());

            Speciate();
        }

        /// <summary>
        /// Creates the phenotypes of the entire population. These networks are used to create a fitness for each genome.
        /// </summary>
        public List<Network> CreatePhenotypes() {
            _phenotypes.Clear();

            foreach (var genome in EntirePopulation) {
                _phenotypes.Add(new Network(genome));
            }

            return _phenotypes;
        }

        /// <summary>
        /// Will create a new population. If species or genomes have been manually altered, call <see cref="Speciate"/> before this!
        /// <br/>
        /// Make sure to set the fitness of every genome!
        /// </summary>
        public void CompleteGeneration() {
            _phenotypes.Clear();

            //get offspring amount for each species
            var offspring = GetOffspringAmount();

            //create new population
            var newPop = new List<Genome>();
            var elites = new List<Genome>();
            foreach (var species in Species.Where(o => o.Members.Count != 0)) {
                //check if species is stagnant
                if (species.IsStagnant(NeatOptions.StagnationThreshold)) continue;

                //remove worst
                species.RemoveWorstMembers(NeatOptions.RemoveWorstPercentage);

                //select new representative by selecting a random member excluding worst members
                //must be called before speciation
                species.Representative = species.Members[_random.Next(species.Members.Count)];

                //copy elite but let it remain in the species for now for creating offspring
                if (species.Members.Count > 5) {
                    var elite = species.Members.MaxBy(o => o.Fitness);
                    elites.Add(elite.Clone());
                }

                //create offspring from remaining genomes in species
                int offspringAmount = offspring.First(o => o.Item1 == species).Item2;
                for (int i = 0; i < offspringAmount; i++) {
                    //either crossover or clone randomly by chance
                    if (_random.NextDouble() <= NeatOptions.CrossoverChance) {
                        var firstGenome = species.RandomByFitness(_random);
                        var secondGenome = species.RandomByFitness(_random);
                        var newGenome = Crossover(firstGenome, secondGenome, _random);
                        newPop.Add(newGenome);
                    } else {
                        newPop.Add(species.RandomByFitness(_random).Clone());
                    }
                }
            }

            //fill population with random genomes from the previous generation
            if (newPop.Count == 0 && elites.Count == 0) {
                for (int i = 0; i < NeatOptions.TargetPopulationSize; i++) {
                    newPop.Add(EntirePopulation[_random.Next(EntirePopulation.Count)]);
                }
            }

            //replace old population
            EntirePopulation = newPop;

            //mutate new population
            foreach (var genome in EntirePopulation) genome.Mutate(InnovationManager, MutationOptions);

            //add elites unchanged
            EntirePopulation.AddRange(elites);

            //speciate population
            Speciate();
        }

        /// <summary>
        /// Speciates the entire population.
        /// </summary>
        private void Speciate() {
            //clear all members of each species
            foreach (var species in Species) species.Members.Clear();

            //reassign each genome to a species
            foreach (var genome in EntirePopulation) {
                //find fitting species
                var species = Species.FirstOrDefault(o => o.Compatible(genome, NeatOptions.GenomeDistanceThreshold), null);

                //if no species found, create new
                if (species == null) {
                    species = new Species(genome);
                    Species.Add(species);
                }

                //add to species
                species.Members.Add(genome);
            }
        }

        /// <summary>
        /// Calculates the new amount of offspring for each species.
        /// Make sure that all genomes fitnesses are adjusted to the species size!
        /// </summary>
        private List<Tuple<Species, int>> GetOffspringAmount() {
            var speciesOffspring = new List<Tuple<Species, int>>();

            double globalAdjustedFitnessSum = Species.Sum(s => s.Members.Sum(m => m.Fitness / s.Members.Count));

            foreach (Species species in Species.Where(o => o.Members.Count != 0)) {
                double localAdjustedFitnessSum = species.Members.Sum(o => o.Fitness / species.Members.Count);

                int eliteCount = Species.Count(o => o.Members.Count > 5);
                int amount = (int)Math.Round(localAdjustedFitnessSum / globalAdjustedFitnessSum * (NeatOptions.TargetPopulationSize - eliteCount));
                
                speciesOffspring.Add(new (species, amount));
            }

            return speciesOffspring;
        }

        private Genome Crossover(Genome genome1, Genome genome2, Random r) {
            Genome newGenome = new Genome();

            var (matching, disjoint, excess) = Core.Species.MatchingDisjointExcess(genome1, genome2);

            //add matching genes from either parent randomly
            foreach (var i in matching) {
                var rndGenome = r.NextDouble() <= 0.5D ? genome1 : genome2;
                var connection = rndGenome.Connections[i];

                newGenome.Connections.Add(i, connection);

                //make sure to update node dictionary so that connections dont point to non-existing nodes
                newGenome.Nodes.TryAdd(connection.StartId, rndGenome.Nodes[connection.StartId]);
                newGenome.Nodes.TryAdd(connection.EndId, rndGenome.Nodes[connection.EndId]);
            }

            //add disjoint and excess only from fittest parent
            foreach (var de in disjoint.Concat(excess)) {

                var fittestGenome = genome1.Fitness >= genome2.Fitness ? genome1 : genome2;

                //only take disjoint and excess from fittest
                if (!fittestGenome.Connections.TryGetValue(de, out var newConnection)) continue;

                newGenome.Connections.Add(de, newConnection);

                //make sure to update node dictionary so that connections dont point to non-existing nodes
                newGenome.Nodes.TryAdd(newConnection.StartId, fittestGenome.Nodes[newConnection.StartId]);
                newGenome.Nodes.TryAdd(newConnection.EndId, fittestGenome.Nodes[newConnection.EndId]);
            }

            //determine enabled status
            foreach (var connection in newGenome.Connections.Values) {
                bool parent1Disabled = genome1.Connections.TryGetValue(connection.Innovation, out var parent1) && !parent1.Enabled;
                bool parent2Disabled = genome2.Connections.TryGetValue(connection.Innovation, out var parent2) && !parent2.Enabled;

                //if one of the parent gene is disabled, determine status by chance
                if (parent1Disabled || parent2Disabled) {
                    var newConnection = connection;
                    newConnection.Enabled = r.NextDouble() <= NeatOptions.EnableChance;
                    newGenome.Connections[connection.Innovation] = newConnection;
                }
            }

            //order all nodes
            newGenome.Nodes = newGenome.Nodes.OrderBy(o => o.Key).ToDictionary();

            return newGenome;
        }
    }
}
