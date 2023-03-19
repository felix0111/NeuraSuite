using System;
using System.Collections.Generic;
using System.Linq;

namespace EasyNNFramework.NEAT {

    [Serializable]
    public class Neat {

        public Dictionary<int, Network> NetworkCollection = new Dictionary<int, Network>();
        public Dictionary<int, Species> Species { get; private set; }

        //innovation number starts at 1
        public Dictionary<(int, int), int> InnovationCollection = new Dictionary<(int, int), int>();

        public Random Random;

        public Neuron[] InputTemplate, ActionTemplate;

        public SpeciationOptions SpeciationOptions;

        private int _speciesCounter;

        public Neat(Neuron[] inputTemplate, Neuron[] actionTemplate, SpeciationOptions speciationOptions) {
            Random = new Random();

            InputTemplate = inputTemplate;
            ActionTemplate = actionTemplate;

            SpeciationOptions = speciationOptions;
            Species = new Dictionary<int, Species>();
        }

        public void AddNetwork(int amount, Network startingNetwork = null, bool migrate = true) {
            for (int i = 0; i < amount; i++) {
                Network newNet;
                if (startingNetwork == null) {
                    newNet = new Network(NetworkCollection.Count, InputTemplate, ActionTemplate);
                    NetworkCollection.Add(NetworkCollection.Count, newNet);
                    
                } else {
                    newNet = migrate ? MigrateNetwork(NetworkCollection.Count, startingNetwork) : startingNetwork;
                    NetworkCollection.Add(NetworkCollection.Count, newNet);
                }
                newNet.ResetNetwork();
            }
            SpeciateAll();
        }

        public void RemoveNetwork(int networkID) {
            if (Species.TryGetValue(NetworkCollection[networkID].SpeciesID, out Species specie)) specie.RemoveFromSpecies(NetworkCollection[networkID]);
            NetworkCollection.Remove(networkID);
        }

        public void RemoveAllNetworks() {
            foreach (var key in NetworkCollection.Keys.ToList()) {
                RemoveNetwork(key);
            }
        }
        
        //after x steps of no fitness improvement => no offspring
        //resets fitness
        public void Repopulate(int targetPopulationSize, bool forcePopulationSize, float mutationChance, int mutationAmount, MutateOptions mutateOptions) {
            
            //linearely maps the avg fitness of a species to a range from 0 to population size
            var speciesPop = SpeciesPopulation(targetPopulationSize);

            if (SpeciationOptions.RemoveUnimproved) speciesPop = speciesPop.Where(o => Species[o.Item1].CheckImprovement(SpeciationOptions.UseAdjustedFitness, SpeciationOptions.StepsUntilUnimprovedDelete)).ToList();

            List<Network> newNetworks = new List<Network>();
            foreach (var speciePop in speciesPop) {

                //linearely maps the network fitness of a species to a range from 0 to the new population size in specie
                var networkPop = Species[speciePop.Item1].PopulationSize(speciePop.Item2, 5);

                foreach (var netPop in networkPop) {
                    for (int i = 0; i < netPop.Item2; i++) newNetworks.Add(new Network(newNetworks.Count, Species[speciePop.Item1].AllNetworks[netPop.Item1]));
                }
            }

            if (newNetworks.Count == 0) newNetworks = NetworkCollection.Select(o => o.Value).Take(targetPopulationSize).ToList();

            //add/remove random networks until at population size
            int count = targetPopulationSize - newNetworks.Count;
            if (!forcePopulationSize) count = 0;
            if (count > 0) {    //fill until at max population size
                for (int i = 0; i < count; i++) {
                    newNetworks.Add(new Network(newNetworks.Count, NetworkCollection.ElementAt(Random.Next(0, NetworkCollection.Count)).Value));
                }
            } else if (count < 0) { //remove until max population size
                for (int i = 0; i < Math.Abs(count); i++) {
                    newNetworks.RemoveAt(Random.Next(0, newNetworks.Count));
                }
            }

            //clear old population
            RemoveAllNetworks();

            //populate all networks
            for (int i = 0; i < newNetworks.Count; i++) {
                if(Random.NextDouble() < mutationChance) for (int j = 0; j < Random.Next(1, mutationAmount+1); j++) newNetworks[i].Mutate(this, mutateOptions);
                AddNetwork(1, newNetworks[i], false);
            }

            SpeciateAll();
            RemoveEmptySpecies();
        }

        public void CalculateAll() {
            for (int i = 0; i < NetworkCollection.Count; i++) {
                NetworkCollection.ElementAt(i).Value.CalculateNetwork();
            }
        }

        public void SpeciateAll() {
            foreach (Network network in NetworkCollection.Values) {
                SpeciateSingle(network);
            }
        }

        public void SpeciateSingle(Network network) {

            //check if (still) compatible to old species, this may improve performance because less compatibility checks are needed
            if (Species.TryGetValue(network.SpeciesID, out Species specie)) {
                if (specie.CheckCompatibility(network, SpeciationOptions, true)) return;

                specie.RemoveFromSpecies(network);
            }

            //find first compatible species and add
            foreach (var species in Species.Values) {
                if (species.CheckCompatibility(network, SpeciationOptions, true)) return;
            }

            //if no compatible found, create new species
            Species newSpecies = new Species(_speciesCounter++, network);
            newSpecies.AddToSpecies(network);
            Species.Add(newSpecies.SpeciesID, newSpecies);
        }

        public void RemoveEmptySpecies() {
            foreach (var specie in Species.Values.ToList()) {
                if (specie.AllNetworks.Count == 0) Species.Remove(specie.SpeciesID);
            }
        }

        //reevaluates the correct innovation IDs for a network which may come from a different population
        //this is because the innovation collection and IDs may not be correct
        public Network MigrateNetwork(int newNetworkID, Network template) {
            Network newNetwork = new Network(newNetworkID, template);

            for (int i = 0; i < newNetwork.Connections.Count; i++) {
                Connection oldConnection = newNetwork.Connections.ElementAt(i).Value;
                newNetwork.RemoveConnection(oldConnection.InnovationID);

                oldConnection.InnovationID = NewInnovation(oldConnection.SourceID, oldConnection.TargetID);
                newNetwork.AddConnection(oldConnection.InnovationID, oldConnection.SourceID, oldConnection.TargetID, oldConnection.Weight);
            }

            for (int i = 0; i < newNetwork.RecurrentConnections.Count; i++) {
                Connection oldConnection = newNetwork.RecurrentConnections.ElementAt(i).Value;
                newNetwork.RemoveConnection(oldConnection.InnovationID);

                oldConnection.InnovationID = NewInnovation(oldConnection.SourceID, oldConnection.TargetID);
                newNetwork.AddConnection(oldConnection.InnovationID, oldConnection.SourceID, oldConnection.TargetID, oldConnection.Weight);
            }

            return newNetwork;
        }

        //if a given connection is not found then a new innovation number is given this connection and stored in the global collection
        public int NewInnovation(int sourceID, int targetID) {
            if (InnovationCollection.TryGetValue((sourceID, targetID), out int ino)) {
                return ino;
            }

            InnovationCollection.Add((sourceID, targetID), InnovationCollection.Count);
            return InnovationCollection.Count-1;
        }

        //returns list of (specieID , newPopSize)
        //when all species fitness is zero, returns empty list
        //limits species amount by the corresponding speciation option, maximum species amount is 1/4 of target population amount
        public List<(int, int)> SpeciesPopulation(int targetAmount) {

            List<(int, float)> newArr = Species.Select(o => (o.Key, o.Value.AverageFitness(SpeciationOptions.UseAdjustedFitness))).OrderByDescending(o => o.Item2).ToList();

            newArr = newArr.Take(Math.Min(targetAmount/4, SpeciationOptions.MaxSpecies)).ToList();
            
            float sum = newArr.Sum(o => o.Item2);
            if (sum == 0) return new List<(int, int)>();

            return newArr.Select(o=> (o.Item1, (int)Math.Ceiling((o.Item2/sum) * targetAmount))).ToList();
        }

    }
}