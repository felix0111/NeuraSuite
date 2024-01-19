using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyNNFramework.NEAT {
    public static class Mutator {

        //returns true if mutated
        public static bool Mutate(this Network network, Neat neat, MutateOptions options) {

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
                if (network.ExistsConnection(rndStart, rndEnd)) return false;
                network.AddConnection(neat.NewInnovation(rndStart, rndEnd), rndStart, rndEnd, NNUtility.RandomWeight(neat.Random));
            } else if (rndChance <= options.AddConnection + options.RemoveConnection) {
                if (network.Connections.Count == 0) return false;
                network.RemoveConnection(network.RandomConnection(neat.Random).InnovationID);
            } else if (rndChance <= options.AddConnection + options.RemoveConnection + options.AddNeuron) {
                if (network.Connections.Count == 0) return false;
                Connection con = network.RandomConnectionType(neat.Random, false);
                Neuron n = network.AddNeuron(neat, con.InnovationID, options.DefaultActivationFunction);
                if (options.RandomDefaultActivationFunction) n.Function = NNUtility.RandomActivationFunction(neat.Random, options.HiddenActivationFunctionPool);
            } else if (rndChance <= options.AddConnection + options.RemoveConnection + options.AddNeuron + options.RemoveNeuron) {
                if (network.HiddenNeurons.Length == 0) return false;
                network.RemoveNeuron(network.HiddenNeurons[neat.Random.Next(0, network.HiddenNeurons.Length)].ID);
            } else if (rndChance <= options.AddConnection + options.RemoveConnection + options.AddNeuron + options.RemoveNeuron + options.RandomFunction) {
                if (network.HiddenNeurons.Length == 0) return false;
                network.HiddenNeurons[neat.Random.Next(0, network.HiddenNeurons.Length)].Function = NNUtility.RandomActivationFunction(neat.Random, options.HiddenActivationFunctionPool);
            } else if (rndChance <= options.AddConnection + options.RemoveConnection + options.AddNeuron + options.RemoveNeuron + options.RandomFunction + options.AddRecurrentConnection) {
                if (network.HiddenNeurons.Length == 0) return false;

                //only allows recurrent connections between hidden neurons
                int counter = 0;
                do {
                    rndStart = network.HiddenNeurons[neat.Random.Next(0, network.HiddenNeurons.Length)].ID;
                    rndEnd = network.HiddenNeurons[neat.Random.Next(0, network.HiddenNeurons.Length)].ID;
                    if (network.CheckRecurrent(rndEnd, rndStart)) {
                        network.AddConnection(neat.NewInnovation(rndEnd, rndStart), rndEnd, rndStart, NNUtility.RandomWeight(neat.Random));
                        break;
                    }
                    
                    counter++;
                } while (counter <= 4);
                if (counter == 5) return false;
            } else if (rndChance <= options.AddConnection + options.RemoveConnection + options.AddNeuron + options.RemoveNeuron + options.RandomFunction + options.AddRecurrentConnection + options.AdjustWeight) {
                if (network.Connections.Count == 0) return false;

                //weight is adjusted by value between -1f and +1f
                float rndSign = NNUtility.RandomSign(neat.Random);
                Connection rndCon = network.RandomConnection(neat.Random);
                network.ChangeWeight(rndCon.InnovationID, NNUtility.Clamp(-4f, 4f, rndCon.Weight + rndSign * (float)neat.Random.NextDouble()));
            } else if (rndChance <= options.AddConnection + options.RemoveConnection + options.AddNeuron + options.RemoveNeuron + options.RandomFunction + options.AddRecurrentConnection + options.AdjustWeight + options.ToggleConnection) {
                if (network.Connections.Count == 0) return false;
                Connection rndCon = network.RandomConnection(neat.Random);
                network.ToggleConnection(rndCon.InnovationID, !rndCon.Activated);
            }

            network.RecalculateStructure();
            return true;
        }

    }

    //defines chances for different mutations
    //must be in percent and add up to 1
    public struct MutateOptions {

        public float AddConnection, RemoveConnection, AdjustWeight, AddRecurrentConnection, ToggleConnection;
        public float AddNeuron, RemoveNeuron, RandomFunction;
        

        //pool is used when mutation creates a new neuron or randomizes a neuron activation function
        public ActivationFunction[] HiddenActivationFunctionPool;

        //if RandomDefaultActivationFunction is true
        //a random function from the specified pool is used for new neurons
        //else DefaultActivationFunction is used
        public ActivationFunction DefaultActivationFunction;
        public bool RandomDefaultActivationFunction;

        public MutateOptions(float addConnection, float removeConnection, float adjustWeight, float toggleConnection, float addNeuron, float removeNeuron, float randomFunction, float addRecurrentConnection, ActivationFunction defaultActivationFunction, ActivationFunction[] hiddenActivationFunctionPool, bool randomDefaultActivationFunction) {
            AddConnection = addConnection;
            RemoveConnection = removeConnection;
            AdjustWeight = adjustWeight;
            ToggleConnection = toggleConnection;
            AddNeuron = addNeuron;
            RemoveNeuron = removeNeuron;
            RandomFunction = randomFunction;
            AddRecurrentConnection = addRecurrentConnection;
            DefaultActivationFunction = defaultActivationFunction;

            HiddenActivationFunctionPool = hiddenActivationFunctionPool;
            RandomDefaultActivationFunction = randomDefaultActivationFunction;
        }

    }
}
