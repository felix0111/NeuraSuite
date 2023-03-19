using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyNNFramework.NEAT {
    public static class Mutator {

        public static void Mutate(this Network network, Neat neat, MutateOptions options) {

            List<Neuron> possibleStartNeurons = new List<Neuron>(network.InputNeurons);
            List<Neuron> possibleEndNeurons = new List<Neuron>(network.ActionNeurons);

            //add hidden neurons if available
            if (network.HiddenNeurons.Length > 0) {
                possibleStartNeurons.AddRange(network.HiddenNeurons);
                possibleEndNeurons.AddRange(network.HiddenNeurons);
            }

            restart:
            //choose random neurons
            int rndStart = possibleStartNeurons[neat.Random.Next(0, possibleStartNeurons.Count)].ID;
            int rndEnd = possibleEndNeurons[neat.Random.Next(0, possibleEndNeurons.Count)].ID;

            if (rndStart == rndEnd) goto restart;

            float rndChance = (float)neat.Random.NextDouble();
            if (rndChance <= options.AddConnection) {
                if (network.CheckRecurrent(rndStart, rndEnd)) (rndStart, rndEnd) = (rndEnd, rndStart);  //if recurrent connection, switch neurons
                network.AddConnection(neat.NewInnovation(rndStart, rndEnd), rndStart, rndEnd, NNUtility.RandomWeight(neat.Random));
            } else if (rndChance <= options.AddConnection + options.RandomizeWeight) {
                if (network.Connections.Count == 0) return; //if no connection, do nothing
                network.ChangeWeight(network.RandomConnection(neat.Random).InnovationID, NNUtility.RandomWeight(neat.Random));
            } else if (rndChance <= options.AddConnection + options.RandomizeWeight + options.RemoveConnection) {
                if (network.Connections.Count == 0) return;
                network.RemoveConnection(network.RandomConnection(neat.Random).InnovationID);
            } else if (rndChance <= options.AddConnection + options.RandomizeWeight + options.RemoveConnection + options.AddNeuron) {
                if (network.Connections.Count == 0) return;
                Connection con = network.RandomConnectionType(neat.Random, false);
                network.AddNeuron(neat, con.InnovationID, options.HiddenActivationFunction);
            } else if (rndChance <= options.AddConnection + options.RandomizeWeight + options.RemoveConnection + options.AddNeuron + options.RemoveNeuron) {
                if (network.HiddenNeurons.Length == 0) return;
                network.RemoveNeuron(network.HiddenNeurons[neat.Random.Next(0, network.HiddenNeurons.Length)].ID);
            } else if (rndChance <= options.AddConnection + options.RandomizeWeight + options.RemoveConnection + options.AddNeuron + options.RemoveNeuron + options.RandomFunction) {
                if (network.HiddenNeurons.Length == 0) return;
                network.HiddenNeurons[neat.Random.Next(0, network.HiddenNeurons.Length)].Function = NNUtility.RandomActivationFunction(neat.Random);
            } else if (rndChance <= options.AddConnection + options.RandomizeWeight + options.RemoveConnection + options.AddNeuron + options.RemoveNeuron + options.RandomFunction + options.AddRecurrentConnection) {
                if (network.HiddenNeurons.Length == 0) return;

                int counter = 0;
                int maxTries = 4;
                while (network.GetNeuronType(rndStart) == NeuronType.Input || network.GetNeuronType(rndStart) == NeuronType.Bias || !network.CheckRecurrent(rndEnd, rndStart)) {
                    rndStart = network.HiddenNeurons[neat.Random.Next(0, network.HiddenNeurons.Length)].ID;
                    rndEnd = possibleEndNeurons[neat.Random.Next(0, possibleEndNeurons.Count)].ID;
                    counter++;
                    if(counter == maxTries) return;
                }

                network.AddConnection(neat.NewInnovation(rndEnd, rndStart), rndEnd, rndStart, NNUtility.RandomWeight(neat.Random));
            } else if (rndChance <= options.AddConnection + options.RandomizeWeight + options.RemoveConnection + options.AddNeuron + options.RemoveNeuron + options.RandomFunction + options.AddRecurrentConnection + options.AdjustWeight) {
                if (network.Connections.Count == 0) return;

                //weight is adjusted by value between -1f and +1f
                float rndSign = NNUtility.RandomSign(neat.Random);
                Connection rndCon = network.RandomConnection(neat.Random);
                network.ChangeWeight(rndCon.InnovationID, NNUtility.Clamp(-4f, 4f, rndCon.Weight + rndSign * (float)neat.Random.NextDouble()));
            }

            network.RecalculateStructure();
        }

    }

    //defines chances for different mutations
    //must be in percent and add up to 1
    public struct MutateOptions {

        public float AddConnection, RandomizeWeight, RemoveConnection, AdjustWeight, AddRecurrentConnection;
        public float AddNeuron, RemoveNeuron, RandomFunction;
        public ActivationFunction HiddenActivationFunction;

        public MutateOptions(float addConnection, float randomizeWeight, float removeConnection, float adjustWeight, float addNeuron, float removeNeuron, float randomFunction, float addRecurrentConnection, ActivationFunction hiddenActivationFunction) {
            AddConnection = addConnection;
            RandomizeWeight = randomizeWeight;
            RemoveConnection = removeConnection;
            AdjustWeight = adjustWeight;
            AddNeuron = addNeuron;
            RemoveNeuron = removeNeuron;
            RandomFunction = randomFunction;
            AddRecurrentConnection = addRecurrentConnection;
            HiddenActivationFunction = hiddenActivationFunction;
        }

    }
}
