﻿using NeuraSuite.Neat.Core;
using NeuraSuite.Neat.Utility;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuraSuite.Neat {
    public class NeatManager {

        public InnovationManager InnovationManager;

        public MutationSettings MutationSettings;
        public SpeciationSettings SpeciationSettings;
        public NeatSettings NeatOptions;

        public List<Genome> EntirePopulation = new();
        public List<Species> Species = new();

        private List<Network> _phenotypes = new();

        private Random _random = new ();

        private int _stagnationCounter;
        private double _bestFitness;

        public NeatManager(Genome initialGenome, NeatSettings neatOptions = default, MutationSettings mutationSettings = default, SpeciationSettings speciationSettings = default) {
            //adjusts innovation and node counter to initial genome
            InnovationManager = new InnovationManager(initialGenome);

            NeatOptions = neatOptions;
            MutationSettings = mutationSettings;
            SpeciationSettings = speciationSettings;
            
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

            //checks if the fitness of the whole population improved
            double bestFitness = EntirePopulation.Max(o => o.Fitness);
            if (bestFitness <= _bestFitness) {
                _stagnationCounter++;
            } else {
                _stagnationCounter = 0;
                _bestFitness = bestFitness;
            }

            //create offspring amount, should be called before removing worst members
            var offspring = GetOffspringAmount();

            //pre process all species
            foreach (var species in Species.Where(o => o.Members.Count != 0)) {
                //remove worst
                species.RemoveWorstMembers(NeatOptions.RemoveWorstPercentage);

                //select new representative by selecting a random member excluding worst members
                //must be called before speciation
                species.Representative = species.Members[_random.Next(species.Members.Count)];
            }

            //create new population
            var newPop = new List<Genome>();
            var elites = new List<Genome>();
            foreach (var (species, offspringAmount) in offspring) {
                //copy elite but let it remain in the species for now for creating offspring
                if (species.Members.Count > 5) {
                    var elite = species.Members.MaxBy(o => o.Fitness);
                    elites.Add(elite.Clone());
                }

                //create offspring from remaining genomes in species
                for (int i = 0; i < offspringAmount; i++) {
                    //either crossover or clone randomly by chance
                    Genome newGenome;
                    if (_random.NextDouble() <= NeatOptions.CrossoverChance) {
                        var firstGenome = species.RandomByFitness(_random);
                        var secondGenome = species.RandomByFitness(_random);
                        newGenome = Crossover(firstGenome, secondGenome, _random);
                    } else {
                        newGenome = species.RandomByFitness(_random).Clone();
                    }

                    newGenome.Mutate(InnovationManager, MutationSettings);
                    newPop.Add(newGenome);
                }
            }

            //if extinction because all species are stagnant
            if (newPop.Count == 0) {
                //get elite of entire (previous) population
                var elite = EntirePopulation.MaxBy(o => o.Fitness);

                //fill population with mutated variants of elite
                for (int i = 0; i < NeatOptions.TargetPopulationSize; i++) {
                    var rndElite = elite.Clone();
                    rndElite.Mutate(InnovationManager, MutationSettings);
                    newPop.Add(rndElite);
                }
            }

            //replace old population
            EntirePopulation = newPop;

            //add elites
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
                var species = Species.FirstOrDefault(o => o.Compatible(genome, SpeciationSettings), null);

                //if no species found, create new
                if (species == null) {
                    species = new Species(genome);
                    Species.Add(species);
                }

                //add to species
                species.Members.Add(genome);
            }

            //remove empty species
            Species.RemoveAll(o => o.Members.Count == 0);
        }

        /// <summary>
        /// Calculates the new amount of offspring for each species.
        /// Make sure that all genomes fitnesses are adjusted to the species size!
        /// </summary>
        private List<Tuple<Species, int>> GetOffspringAmount() {
            var speciesOffspring = new List<Tuple<Species, int>>();

            //calculate the average fitness of all species
            var averageFitnesses = Species.Where(o => o.Members.Count > 0 && !o.IsStagnant(NeatOptions.SpeciesStagnationThreshold)).ToDictionary(species => species, species => species.AverageFitness);

            //if the whole population is not improving, only take top 2 species
            if (_stagnationCounter >= NeatOptions.PopulationStagnationThreshold) averageFitnesses = averageFitnesses.OrderByDescending(o => o.Value).Take(2).ToDictionary();

            //sums the average fitness of all species that produce offspring
            double averageFitnessSum = averageFitnesses.Values.Sum();

            //calculate amount of offspring from average fitness
            foreach (var pair in averageFitnesses) {
                //calculates the share of offspring this species will get
                double populationShare = pair.Value / averageFitnessSum;

                int eliteCount = Species.Count(o => o.Members.Count > 5);
                int amount = (int)Math.Round(populationShare * (NeatOptions.TargetPopulationSize - eliteCount));
                
                speciesOffspring.Add(new (pair.Key, amount));
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
