using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Permissions;
using System.Text;
using System.Threading.Tasks;

namespace EasyNNFramework.NEAT {

    [Serializable]
    public class Species {

        public Network Representative;

        public readonly int SpeciesID;

        public Dictionary<int, Network> AllNetworks;

        private float _bestAverageFitness = -1f;
        public int StepsSinceImprovement { get; private set; }

        public Species(int speciesId, Network representative) {
            Representative = new Network(-1, representative);

            SpeciesID = speciesId;

            AllNetworks = new Dictionary<int, Network>();
        }

        public bool CheckCompatibility(Network network, SpeciationOptions options, bool addToSpecies) {
            bool comp = NEATUtility.Distance(Representative, network, options) < options.CompatabilityThreshold;

            if (addToSpecies && comp) AddToSpecies(network);
            return comp;
        }

        public bool AddToSpecies(Network network) {
            network.SpeciesID = SpeciesID;

            if (AllNetworks.ContainsKey(network.NetworkID)) return false;
            AllNetworks.Add(network.NetworkID, network);
            return true;
        }

        public bool RemoveFromSpecies(Network network) {
            network.SpeciesID = -1;
            return AllNetworks.Remove(network.NetworkID);
        }

        //basically the average of all networks' (adjusted) fitness
        public float AverageFitness(bool useAdjFitness) {

            float val = 0;
            foreach (var network in AllNetworks) {
                val += useAdjFitness ? network.Value.Fitness / AllNetworks.Count : network.Value.Fitness;
            }
            val = val / AllNetworks.Count;

            if (float.IsNaN(val)) val = 0;
            
            return val;
        }

        public bool CheckImprovement(bool useAdjFitness) {
            float newAvgFitness = AverageFitness(useAdjFitness);
            if (_bestAverageFitness >= newAvgFitness) { //no improvement
                StepsSinceImprovement++;
                return false;
            } else {
                _bestAverageFitness = newAvgFitness;
                StepsSinceImprovement = 0;  //improvement
                return true;
            }
        }

        public bool CheckImprovement(bool useAdjFitness, int steps) {
            float newAvgFitness = AverageFitness(useAdjFitness);
            if (_bestAverageFitness >= newAvgFitness) { //no improvement
                StepsSinceImprovement++;
            } else {
                _bestAverageFitness = newAvgFitness;
                StepsSinceImprovement = 0;  //improvement
            }

            return StepsSinceImprovement <= steps;
        }

        //returns empty when every fitness is 0
        //values lie between 0 and 1 and add up to 1
        //variety of networks returned is determined by eliteCount, maxmimum is target population size
        public List<(int, int)> PopulationSize(int targetPopulationSize, int eliteCount) {

            List<(int, int)> newArr = new List<(int, int)>();
            var orderedAndReduced = AllNetworks.OrderByDescending(o => o.Value.Fitness).Take(Math.Min(targetPopulationSize, eliteCount)).ToArray();

            float fitnessSum = orderedAndReduced.Sum(o => o.Value.Fitness);
            if (fitnessSum == 0) return newArr;

            foreach (var network in orderedAndReduced) {
                newArr.Add((network.Key, (int) Math.Round((network.Value.Fitness / fitnessSum) * targetPopulationSize)));
            }

            return newArr;
        }
    }

    public struct SpeciationOptions {
        public float DisjointFactor, WeightFactor, CompatabilityThreshold;

        public int MaxSpecies;

        public bool RemoveUnimproved;
        public int StepsUntilUnimprovedDelete;

        public bool UseAdjustedFitness;

        //compatability threshold - higher number means more differences allowed
        //maxSpecies - the more species are allowed, the more lower performing specie can evolve.
        public SpeciationOptions(float disjointFactor, float weightFactor, float compatabilityThreshold, int maxSpecies, bool useAdjustedFitness, bool removeUnimproved, int stepsUntilUnimprovedDelete) {
            DisjointFactor = disjointFactor;
            WeightFactor = weightFactor;
            CompatabilityThreshold = compatabilityThreshold;
            MaxSpecies = maxSpecies;
            UseAdjustedFitness = useAdjustedFitness;
            RemoveUnimproved = removeUnimproved;
            StepsUntilUnimprovedDelete = stepsUntilUnimprovedDelete;
        }
    }
}
