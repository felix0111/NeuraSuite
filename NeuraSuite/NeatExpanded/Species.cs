using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Permissions;
using System.Text;
using System.Threading.Tasks;

namespace NeuraSuite.NeatExpanded {

    [Serializable]
    public class Species {

        /// <summary>
        /// The network this species is based upon. Every network in this species should be compatible to the representative.
        /// </summary>
        public Network Representative;

        public readonly int SpeciesID;

        public Dictionary<int, Network> AllNetworks;

        public float BestAverageFitness = -1f;

        public Species(int speciesId, Network representative) {
            Representative = new Network(-1, representative);

            SpeciesID = speciesId;
            
            AllNetworks = new Dictionary<int, Network>();
        }

        /// <summary>
        /// Checks if a network is compatible to this species.
        /// </summary>
        /// <param name="addToSpecies">If true, the network will be automatically added to this species if compatible.</param>
        public bool CheckCompatibility(Network network, SpeciationOptions options, bool addToSpecies) {
            bool comp = NEATUtility.Distance(Representative, network, options) <= options.CompatabilityThreshold;

            if (addToSpecies && comp) AddToSpecies(network);
            return comp;
        }

        /// <summary>
        /// Adds a network to this species. You should normally <see cref="CheckCompatibility"/> beforehand!
        /// </summary>
        public bool AddToSpecies(Network network) {
            network.SpeciesID = SpeciesID;

            if (AllNetworks.ContainsKey(network.NetworkID)) return false;
            AllNetworks.Add(network.NetworkID, network);
            return true;
        }

        /// <summary>
        /// Removes a network from this species.
        /// </summary>
        public bool RemoveFromSpecies(Network network) {
            network.SpeciesID = -1;
            return AllNetworks.Remove(network.NetworkID);
        }

        /// <summary>
        /// Calculates the average fitness of all networks in this species.
        /// </summary>
        /// <param name="useAdjFitness">If true, penalizes larger species and gives smaller ones an advantage.</param>
        /// <param name="median">If true, might filter out statistical outliers much better.</param>
        public float AverageFitness(bool useAdjFitness, bool median) {
            if (AllNetworks.Count == 0) return 0f;

            float val = 0;
            if (median) {
                //calculate median, order by fitness and select the network in the middle
                val = AllNetworks.OrderByDescending(o => o.Value.Fitness).ElementAt( (int)Math.Ceiling(AllNetworks.Count / 2f) - 1).Value.Fitness;
                if(useAdjFitness) val /= AllNetworks.Count;
            } else {
                val = AllNetworks.Average(o => useAdjFitness ? o.Value.Fitness / AllNetworks.Count : o.Value.Fitness);
            }

            if(val > BestAverageFitness) BestAverageFitness = val;  //update best average fitness of this species

            return val;
        }

        /// <summary>
        /// Used to create a population with only the best x networks of this species where the amount of different networks is defined by <see cref="networkVariety"/>.
        /// <br/>
        /// When every fitness is 0, returns empty list otherwise returns a list of tuples: (network ID, amount)
        /// </summary>
        /// <param name="targetPopulationSize">The total amount of networks in the new population.</param>
        /// <param name="networkVariety">
        /// How many of the first best networks are used to create the new population.
        /// Basically determines the variety of networks in the new population.
        /// </param>
        public List<(int, int)> CreateSpeciesPopulation(int targetPopulationSize, int networkVariety) {

            //take the best x networks with x = eliteCount (maximum is targetSpeciesSize)
            List<(int, int)> newArr = new List<(int, int)>();
            var orderedAndReduced = AllNetworks.OrderByDescending(o => o.Value.Fitness).Take(Math.Min(targetPopulationSize, networkVariety)).ToArray();

            //sum fitness of all selected networks
            float fitnessSum = orderedAndReduced.Sum(o => o.Value.Fitness);
            if (fitnessSum == 0) return newArr;

            //linearly scales the fitness of each selected network to the targetSpeciesSize
            foreach (var network in orderedAndReduced) {
                newArr.Add((network.Key, (int) Math.Round((network.Value.Fitness / fitnessSum) * targetPopulationSize)));
            }

            return newArr;
        }

        /// <summary>
        /// Gets a random member. Networks with higher fitness have a higher chance.
        /// </summary>
        public Network RandomByFitness(Random r) {
            if (AllNetworks.Count == 0) return null;

            var fitnessSum = AllNetworks.Values.Sum(o => o.Fitness);
            var normalizedFitnesses = AllNetworks.Values.Select(o => (o, o.Fitness / fitnessSum)).OrderBy(o => o.Item2).ToList();

            double rnd = r.NextDouble();
            double probabilitySum = 0D;
            foreach (var genome in normalizedFitnesses) {
                probabilitySum += genome.Item2;
                if (rnd <= probabilitySum) return genome.o;
            }

            throw new Exception("Should not happen!");
        }
    }

    /// <summary>
    /// Options for defining the behaviour when assigning networks to species.
    /// </summary>
    public struct SpeciationOptions {

        /// <summary>
        /// If the distance of two networks are smaller or equal than this threshold, then they belong to one species.
        /// <br/> <br/>
        /// A value of 0.3 is suggested. A higher threshold may decrease the amount of species.
        /// </summary>
        public float CompatabilityThreshold;

        /// <summary>
        /// Determines how much influence the amount of disjoints over connection amount has on species compatability.
        /// <br/> <br/>
        /// A value of 1 is suggested. A higher value may increase the amount of individual species.
        /// </summary>
        public float DisjointFactor;

        /// <summary>
        /// Determines how much influence the absolute difference in weights has on species compatability.
        /// <br/> <br/>
        /// A value of 0 is suggested. A value higher than 0.1 may increase the amount of individual species drastically.
        /// Counteract by increasing <see cref="CompatabilityThreshold"/>!
        /// </summary>
        public float WeightFactor;

        /// <summary>
        /// When true, penalizes larger species and gives smaller ones an advantage.
        /// </summary>
        // TODO move variable to another place?
        public bool UseAdjustedFitness;

        /// <param name="disjointFactor"><inheritdoc cref="DisjointFactor"/></param>
        /// <param name="weightFactor"><inheritdoc cref="WeightFactor"/></param>
        /// <param name="compatabilityThreshold"><inheritdoc cref="CompatabilityThreshold"/></param>
        /// <param name="useAdjustedFitness"><inheritdoc cref="UseAdjustedFitness"/></param>
        public SpeciationOptions(float disjointFactor = 1f, float weightFactor = 0f, float compatabilityThreshold = 0.3f, bool useAdjustedFitness = false) {
            DisjointFactor = disjointFactor;
            WeightFactor = weightFactor;
            CompatabilityThreshold = compatabilityThreshold;
            UseAdjustedFitness = useAdjustedFitness;
        }
    }
}
