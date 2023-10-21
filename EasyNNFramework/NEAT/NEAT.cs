using System;
using System.Collections.Generic;
using System.Linq;
using Random = System.Random;

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

        public Network AddNetwork(int ID) {
            Network newNet = new Network(ID, InputTemplate, ActionTemplate);
            NetworkCollection.Add(newNet.NetworkID, newNet);
            
            return newNet;
        }

        //change the structure of an existing network and speciates it
        public void ChangeNetwork(int networkID, Network template) {
            if (Species.TryGetValue(NetworkCollection[networkID].SpeciesID, out Species specie)) specie.RemoveFromSpecies(NetworkCollection[networkID]);

            NetworkCollection[networkID] = MigrateNetwork(networkID, template);

            SpeciateSingle(NetworkCollection[networkID]);
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

            //transfer improvement data to new species
            if (oldSpecie != null) {
                newSpecies.StepsSinceImprovement = oldSpecie.StepsSinceImprovement;
                newSpecies.BestAverageFitness = oldSpecie.BestAverageFitness;
                
            }

            newSpecies.AddToSpecies(network);
            Species.Add(newSpecies.SpeciesID, newSpecies);
        }

        public void AdjustCompatabilityFactor(int currentSpeciesAmount, float step) {
            if (SpeciationOptions.MaxSpecies == currentSpeciesAmount) return;

            var oldOptions = SpeciationOptions;
            float adj = currentSpeciesAmount < SpeciationOptions.MaxSpecies ? -step : step;

            oldOptions.CompatabilityThreshold += adj;
            SpeciationOptions = oldOptions;
        }

        public void RemoveEmptySpecies() {
            foreach (var specie in Species.Values.ToList()) {
                if (specie.AllNetworks.Count == 0) Species.Remove(specie.SpeciesID);
            }
        }

        //doesnt work for different input/action neurons
        //reevaluates the correct innovation IDs for a network which may come from a different population
        //this is because the innovation collection and IDs may not be correct
        public Network MigrateNetwork(int newNetworkID, Network template) {
            Network newNetwork = new Network(newNetworkID, template);

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
        //spreadfactor defines how much more population a better species gets; 1 = linear spread over all species
        public List<(int, int)> SpeciesPopulation(int targetNetworkAmount, int spreadFactor) {

            List<(int, float)> newArr = Species.Select(o => (o.Key, o.Value.AverageFitness(SpeciationOptions.UseAdjustedFitness))).OrderByDescending(o => o.Item2).ToList();

            if(SpeciationOptions.RemoveUnimproved) newArr = newArr.Where(o => Species[o.Item1].ImprovedSince(SpeciationOptions.UseAdjustedFitness, SpeciationOptions.StepsUntilUnimprovedDelete)).ToList();

            newArr = newArr.Take(Math.Min(targetNetworkAmount / 4, SpeciationOptions.MaxSpecies)).ToList();

            //if not softmax, spread linearly
            float sum = newArr.Sum(o => (float)Math.Pow(o.Item2, spreadFactor));
            if (sum == 0) return new List<(int, int)>();
            return newArr.Select(o=> (o.Item1, (int)Math.Ceiling((Math.Pow(o.Item2, spreadFactor)/sum) * targetNetworkAmount))).ToList();
        }

    }
}