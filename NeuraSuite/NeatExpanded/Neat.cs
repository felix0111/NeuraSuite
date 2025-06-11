using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Random = System.Random;

namespace NeuraSuite.NeatExpanded {

    [Serializable]
    public class Neat {

        public Random Random;
        
        public Dictionary<int, Network> NetworkCollection = new Dictionary<int, Network>();
        public Dictionary<int, Species> Species = new Dictionary<int, Species>();

        /// <summary>
        /// Stores connection information.
        /// Innovation identifier starts at 0.
        /// <br/> <br/>
        /// Dictionary structure: KEY=(source neuron ID, target neuron ID) VALUE=innovation ID
        /// </summary>
        // TODO why not innovation ID as dict key?
        public Dictionary<(int, int), int> InnovationCollection = new Dictionary<(int, int), int>();

        /// <summary>
        /// Templates for input/output neurons.
        /// </summary>
        public Neuron[] InputTemplate, ActionTemplate;

        public SpeciationOptions SpeciationOptions;
        private int _speciesCounter;

        public int GetNeuronId => _neuronIdCounter++;
        private int _neuronIdCounter;

        /// <summary>
        /// The <see cref="InputTemplate"/> and <see cref="ActionTemplate"/> is used for all networks created in this population.
        /// </summary>
        public Neat(Neuron[] inputTemplate, Neuron[] actionTemplate, SpeciationOptions speciationOptions) {
            Random = new Random();

            InputTemplate = inputTemplate;
            ActionTemplate = actionTemplate;
            
            SpeciationOptions = speciationOptions;

            _neuronIdCounter = Math.Max(inputTemplate.Max(o => o.ID), actionTemplate.Max(o => o.ID)) + 1;
        }

        /// <summary>
        /// Adds a new empty network to the population.
        /// </summary>
        public Network AddNetwork(int ID) {
            Network newNet = new Network(ID, InputTemplate, ActionTemplate);
            NetworkCollection.Add(newNet.NetworkID, newNet);
            
            return newNet;
        }

        /// <summary>
        /// Adds a new network to the population.
        /// </summary>
        public Network AddNetwork(int ID, Network template) {
            var network = AddNetwork(ID);
            ChangeNetwork(ID, template);

            return network;
        }

        /// <summary>
        /// Changes the structure of an existing network and speciates it.
        /// </summary>
        public void ChangeNetwork(int networkID, Network template) {
            if (Species.TryGetValue(NetworkCollection[networkID].SpeciesID, out Species specie)) specie.RemoveFromSpecies(NetworkCollection[networkID]);
            NetworkCollection[networkID] = MigrateNetwork(networkID, template);

            SpeciateSingle(NetworkCollection[networkID]);
        }

        /// <summary>
        /// Removes a single network from the population.
        /// </summary>
        public void RemoveNetwork(int networkID) {
            if (Species.TryGetValue(NetworkCollection[networkID].SpeciesID, out Species specie)) specie.RemoveFromSpecies(NetworkCollection[networkID]);
            NetworkCollection.Remove(networkID);
        }

        /// <summary>
        /// Removes all networks.
        /// </summary>
        public void RemoveAllNetworks() {
            foreach (var key in NetworkCollection.Keys.ToList()) {
                RemoveNetwork(key);
            }
        }

        /// <summary>
        /// Calculates all networks.
        /// </summary>
        public void CalculateAll() {
            for (int i = 0; i < NetworkCollection.Count; i++) {
                NetworkCollection.ElementAt(i).Value.CalculateNetwork();
            }
        }

        /// <summary>
        /// Assigns all networks to a species
        /// </summary>
        public void SpeciateAll() {
            foreach (Network network in NetworkCollection.Values) {
                SpeciateSingle(network);
            }
        }

        /// <summary>
        /// Assigns a single network to a species.
        /// </summary>
        /// <param name="network"></param>
        public void SpeciateSingle(Network network) {

            //check if still compatible to old species, this may improve performance because less compatibility checks are needed
            if (Species.TryGetValue(network.SpeciesID, out Species oldSpecie)) {
                if (oldSpecie.CheckCompatibility(network, SpeciationOptions, true)) return;

                oldSpecie.RemoveFromSpecies(network);
            }

            //find first compatible species and add
            foreach (var species in Species.Values) {
                if (species.CheckCompatibility(network, SpeciationOptions, true)) return;
            }

            //if no compatible found, create new species
            Species newSpecies = new Species(_speciesCounter++, network);
            newSpecies.BestAverageFitness = network.Fitness;

            newSpecies.AddToSpecies(network);
            Species.Add(newSpecies.SpeciesID, newSpecies);
        }

        /// <summary>
        /// This function will try to dynamically adjust <see cref="Neat.SpeciationOptions.CompatabilityThreshold"/> so that the amount of species matches <see cref="targetSpeciesAmount"/>.
        /// <br/>
        /// 
        /// </summary>
        /// <param name="step">
        /// How much the <see cref="Neat.SpeciationOptions.CompatabilityThreshold"/> changes with each adjustment.
        /// Using a fraction of <see cref="Neat.SpeciationOptions.DisjointFactor"/> is suggested.
        /// <br/>
        /// When <see cref="Neat.SpeciationOptions.WeightFactor"/> is higher than 0, you may have to increase the step size significantly.
        /// </param>
        // TODO change step size by difference and MaxSpecies
        public void AdjustCompatabilityFactor(float step, int targetSpeciesAmount) {
            int fullSpecies = Species.Count(o => o.Value.AllNetworks.Count != 0);

            if (targetSpeciesAmount == fullSpecies) return;
            var oldOptions = SpeciationOptions;
            float adj = fullSpecies < targetSpeciesAmount ? -step : step;

            //clamp incase the compatability threshold increases/decreases too much
            oldOptions.CompatabilityThreshold += Utility.Clamp(0.001f, float.MaxValue, adj);
            SpeciationOptions = oldOptions;
        }

        /// <summary>
        /// Removes all species which contain no networks. This is not always necessary.
        /// </summary>
        public void RemoveEmptySpecies() {
            foreach (var specie in Species.Values.ToList()) {
                if (specie.AllNetworks.Count == 0) Species.Remove(specie.SpeciesID);
            }
        }

        /// <summary>
        /// Reevaluates the correct innovation IDs for a network which may come from a different population and returns the new one.
        /// <br/>
        /// Does not work when input/output neurons are different from <see cref="InputTemplate"/> and <see cref="ActionTemplate"/>!
        /// </summary>
        /// <exception cref="Exception">Input/Output neurons do not match the neurons of this population.</exception>
        public Network MigrateNetwork(int newNetworkID, Network template) {
            //check if input/output neurons match
            if (template.InputNeurons.Length != InputTemplate.Length || template.ActionNeurons.Length != ActionTemplate.Length) 
                throw new Exception("Input/Output neurons of template network do not match the neurons of this population!");

            Network newNetwork = new Network(newNetworkID, template);

            //needed when temporarily removing connections, so that neurons dont get removed
            newNetwork.AllowUselessHidden = true;

            //remove connection and reevaluate the correct innovation identifier
            List<Connection> newConns = new List<Connection>();
            foreach (var connection in newNetwork.Connections.Values.ToList()) {
                newNetwork.RemoveConnection(connection.InnovationID);
                int newInnov = NewInnovation(connection.SourceID, connection.TargetID);
                newConns.Add(new Connection(newInnov, connection.SourceID, connection.TargetID, connection.Weight));
            }
            foreach (var connection in newNetwork.RecurrentConnections.Values.ToList()) {
                newNetwork.RemoveConnection(connection.InnovationID);
                int newInnov = NewInnovation(connection.SourceID, connection.TargetID);
                newConns.Add(new Connection(newInnov, connection.SourceID, connection.TargetID, connection.Weight));
            }

            foreach (var connection in newConns) {
                newNetwork.AddConnection(connection.InnovationID, connection.SourceID, connection.TargetID, connection.Weight);
            }

            //set AllowUselessHidden to the value of the template
            newNetwork.AllowUselessHidden = template.AllowUselessHidden;

            return newNetwork;
        }

        /// <summary>
        /// Crossover two networks according to the NEAT algorithm.
        /// </summary>
        /// <exception cref="Exception">Input/Output neurons of networks not matching.</exception>
        public Network CrossoverNetworks(int newNetworkID, Network network1, Network network2) {
            if (network1.InputNeurons.Length != network2.InputNeurons.Length || network1.ActionNeurons.Length != network2.ActionNeurons.Length) throw new Exception("Input and Output neurons are not matching!");

            //use fittest as template
            var fittest = network1.Fitness > network2.Fitness ? network1 : network2;
            Network newNetwork = new Network(newNetworkID, fittest);

            //matching are randomly chosen from either parent
            var differences = NEATUtility.GetMatchingAndDisjoint(network1, network2);
            foreach (var matchingInnov in differences.Item1) {
                var rndNetwork = Random.NextDouble() < 0.5D ? network1 : network2;
                Connection c = rndNetwork.Connections.TryGetValue(matchingInnov, out Connection con) ? con : rndNetwork.RecurrentConnections[matchingInnov];

                //check if normal or recurrent
                if (newNetwork.Connections.ContainsKey(matchingInnov)) {
                    newNetwork.Connections[matchingInnov] = c;
                } else {
                    newNetwork.RecurrentConnections[matchingInnov] = c;
                }
            }

            //determine enabled status
            foreach (var connection in newNetwork.Connections.Values) {
                bool parent1Disabled = network1.Connections.TryGetValue(connection.InnovationID, out var parent1) && !parent1.Activated;
                bool parent2Disabled = network2.Connections.TryGetValue(connection.InnovationID, out var parent2) && !parent2.Activated;

                //if one of the parent gene is disabled, determine status by chance
                if (parent1Disabled || parent2Disabled) {
                    var newConnection = connection;
                    newConnection.Activated = Random.NextDouble() <= 0.25D;
                    newNetwork.Connections[connection.InnovationID] = newConnection;
                }
            }

            return newNetwork;
        }

        /// <summary>
        /// If a connection between the two neurons is not found then a new innovation number is returned and stored in <see cref="InnovationCollection"/>.
        /// </summary>
        public int NewInnovation(int sourceID, int targetID) {
            if (InnovationCollection.TryGetValue((sourceID, targetID), out int ino)) {
                return ino;
            }

            InnovationCollection.Add((sourceID, targetID), InnovationCollection.Count);
            return InnovationCollection.Count-1;
        }

        /// <summary>
        /// Returns a list of tuple: (speciesID, newPopSize). This is used to determine the size of each species in a new population.
        /// <see cref="Neat.SpeciationOptions.MaxSpeciesPerPopulation"/> determines the maximum variety of species in the new population.
        /// <br/>
        /// Each network must have a <see cref="Network.Fitness"/> value.
        /// </summary>
        /// <param name="targetNetworkAmount">The targeted total amount of networks in all species.</param>
        /// <param name="spreadFactor">
        /// spreadFactor = 1, species fitness is linear scaled to targetNetworkAmount <br/>
        /// spreadFactor &gt; 1, species with higher fitness are much bigger compared to species with lower fitness <br/>
        /// spreadFactor &lt; 1, species with higher fitness have a smaller amount-wise advantage compared to species with lower fitness <br/>
        /// </param>
        // TODO use median instead of average?
        public List<(int, int)> CreatePopulation(int targetNetworkAmount, int spreadFactor) {
            //calculate the average fitness of each species
            List<(int, float)> speciesFitnessPair = Species.Select(o => (o.Key, o.Value.AverageFitness(SpeciationOptions.UseAdjustedFitness, false))).OrderByDescending(o => o.Item2).ToList();
            
            //only take the best x species determined by MaxSpecies parameter
            speciesFitnessPair = speciesFitnessPair.Take(SpeciationOptions.MaxSpeciesPerPopulation).ToList();

            //calculate the sum of the network distribution factors, used for normalizing the distribution factors
            float sum = speciesFitnessPair.Sum(o => (float)Math.Pow(o.Item2, spreadFactor));

            //if all species have a average fitness of 0, equal distribution
            if (sum == 0) return speciesFitnessPair.Select(o => (o.Item1, targetNetworkAmount / speciesFitnessPair.Count)).ToList();

            //normalize network distribution factor to values 0-1, so we can multiply by targetNetworkAmount
            //normalization formula: fitness^spread / sum(fitness^spread)
            //tuple structure: (speciesID, normalized network dist. factor * targetNetworkAmount)
            List<(int, int)> result = new List<(int, int)>();
            foreach (var tuple in speciesFitnessPair) {
                int newAmount = (int)(Math.Pow(tuple.Item2, spreadFactor) / sum * targetNetworkAmount);
                result.Add((tuple.Item1, newAmount));
            }

            return result;
        }

        public List<(int, int)> GetOffspringAmount(int targetPopulationSize) {
            var offspring = new List<(int, int)>();

            double globalFitnessSum = Species.Values.Sum(o => o.AllNetworks.Values.Sum(n => SpeciationOptions.UseAdjustedFitness ? n.Fitness / o.AllNetworks.Count : n.Fitness));

            foreach (var species in Species.Values) {
                double localFitnessSum = species.AllNetworks.Values.Sum(o => SpeciationOptions.UseAdjustedFitness ? o.Fitness / species.AllNetworks.Count : o.Fitness);

                int eliteCount = Species.Values.Count(o => o.AllNetworks.Count > 5);
                int amount = (int)Math.Round(localFitnessSum / globalFitnessSum * (targetPopulationSize - eliteCount));

                offspring.Add((species.SpeciesID, amount));
            }

            return offspring;
        }


        public void CompleteGeneration(int targetPopulationSize, double removeWorstPercentage, double crossoverChance, MutateOptions options) {
            var offspring = GetOffspringAmount(targetPopulationSize);

            //create new population
            var newPop = new List<Network>();
            var elites = new List<Network>();
            foreach (var species in Species.Values.Where(o => o.AllNetworks.Count != 0)) {
                //check if species is stagnant
                //if (species.IsStagnant(NeatOptions.StagnationThreshold)) continue;

                //remove worst
                int worstAmount = (int)Math.Round(species.AllNetworks.Values.Count * removeWorstPercentage);
                var worst = species.AllNetworks.Values.OrderBy(o => o.Fitness).Take(worstAmount);
                foreach (var network in worst) species.RemoveFromSpecies(network);

                //select new representative by selecting a random member excluding worst members
                //must be called before speciation
                species.Representative = new Network(-1, species.AllNetworks.Values.ToArray()[Random.Next(species.AllNetworks.Count)]);

                //copy elite but let it remain in the species for now for creating offspring
                if (species.AllNetworks.Count > 5) {
                    var elite = species.AllNetworks.Values.MaxBy(o => o.Fitness);
                    elites.Add(new Network(elites.Count + newPop.Count, elite));
                }

                //create offspring from remaining networks in species
                int offspringAmount = offspring.First(o => o.Item1 == species.SpeciesID).Item2;
                for (int i = 0; i < offspringAmount; i++) {
                    //either crossover or clone randomly by chance
                    if (Random.NextDouble() <= crossoverChance) {
                        var firstGenome = species.RandomByFitness(Random);
                        var secondGenome = species.RandomByFitness(Random);
                        var newGenome = CrossoverNetworks(elites.Count + newPop.Count, firstGenome, secondGenome);
                        newPop.Add(newGenome);
                    } else {
                        newPop.Add(new Network(elites.Count + newPop.Count, species.RandomByFitness(Random)));
                    }
                }

                species.AllNetworks.Clear();
            }

            //fill population placeholder with random networks from the previous generation
            if (newPop.Count == 0 && elites.Count == 0) {
                for (int i = 0; i < targetPopulationSize; i++) {
                    newPop.Add(new Network(i, NetworkCollection.Values.ToArray()[Random.Next(NetworkCollection.Count)]));
                }
            }

            //clear old population
            NetworkCollection.Clear();

            //create new population
            foreach (var network in newPop) NetworkCollection.Add(network.NetworkID, network);

            //mutate new population
            foreach (var network in NetworkCollection.Values) network.Mutate(this, options);

            //add elites unchanged
            foreach (var network in elites) {
                NetworkCollection.Add(network.NetworkID, network);
            }

            //speciate population
            SpeciateAll();
        }
    }
}