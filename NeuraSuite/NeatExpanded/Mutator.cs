using System.Collections.Generic;

namespace NeuraSuite.NeatExpanded {
    public static class Mutator {

        //returns true if mutated
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
            
            //Add connection
            if (neat.Random.NextDouble() <= options.AddConnection) {
                if (!network.ExistsConnection(rndStart, rndEnd)) {
                    network.AddConnection(neat.NewInnovation(rndStart, rndEnd), rndStart, rndEnd, (float)neat.Random.RandomWeight());
                }
            }
            
            //Remove connection
            if (neat.Random.NextDouble() <= options.RemoveConnection && network.Connections.Count != 0) {
                network.RemoveConnection(network.RandomConnection(neat.Random).InnovationID);
            }
            
            //Add neuron
            if (neat.Random.NextDouble() <= options.AddNeuron && network.Connections.Count != 0) {
                Connection con = network.RandomConnectionType(neat.Random, false);
                Neuron n = network.AddNeuron(neat, con.InnovationID, options.DefaultActivationFunction);
                if (options.RandomDefaultActivationFunction) n.Function = Utility.RandomActivationFunction(neat.Random, options.HiddenActivationFunctionPool);
            }
            
            //Remove neuron
            if (neat.Random.NextDouble() <= options.RemoveNeuron && network.HiddenNeurons.Length != 0) {
                network.RemoveNeuron(network.HiddenNeurons[neat.Random.Next(0, network.HiddenNeurons.Length)].ID);
            }
            
            //Random function
            if (neat.Random.NextDouble() <= options.RandomFunction && network.HiddenNeurons.Length != 0) {
                network.HiddenNeurons[neat.Random.Next(0, network.HiddenNeurons.Length)].Function = Utility.RandomActivationFunction(neat.Random, options.HiddenActivationFunctionPool);
            }

            //mutation of each weight
            foreach (var connection in network.Connections.Values) {
                //Adjust weight
                if (neat.Random.NextDouble() <= options.AdjustWeight && network.Connections.Count != 0) {
                    //weight is adjusted by value between -1f and +1f with the resulting weight clamped between -4f and 4f
                    network.ChangeWeight(connection.InnovationID, Utility.Clamp(-4f, 4f, connection.Weight + (float)neat.Random.RandomWeight(options.WeightAdjustmentMagnitude)));
                }

                //Toggle connection
                if (neat.Random.NextDouble() <= options.ToggleConnection && network.Connections.Count != 0) {
                    network.ToggleConnection(connection.InnovationID, !connection.Activated);
                }
            }

            network.RecalculateStructure(!network.AllowUselessHidden);
        }

    }

    //defines chances for different mutations
    //must be a value between 0 and 1
    public struct MutateOptions {

        public double AddConnection, RemoveConnection, AdjustWeight, ToggleConnection;
        public double AddNeuron, RemoveNeuron, RandomFunction;

        /// <summary>
        /// Specifies the maximum magnitude a weight of a connection can change by mutation.
        /// </summary>
        public double WeightAdjustmentMagnitude;

        /// <summary>
        /// Everytime a mutation creates a new neuron, the activation function is randomly selected from this pool.
        /// <br/>
        /// A random activation function is only selected when <see cref="RandomDefaultActivationFunction"/> is true!
        /// Else <see cref="DefaultActivationFunction"/> is used.
        /// </summary>
        public ActivationFunction[] HiddenActivationFunctionPool;

        //if RandomDefaultActivationFunction is true
        //a random function from the specified pool is used for new neurons
        //else DefaultActivationFunction is used
        public ActivationFunction DefaultActivationFunction;
        public bool RandomDefaultActivationFunction;

        public MutateOptions(double addConnection, double removeConnection, double adjustWeight, double toggleConnection, double addNeuron, double removeNeuron, double randomFunction, double weightAdjustmentMagnitude, ActivationFunction defaultActivationFunction, ActivationFunction[] hiddenActivationFunctionPool, bool randomDefaultActivationFunction) {
            AddConnection = addConnection;
            RemoveConnection = removeConnection;
            AdjustWeight = adjustWeight;
            ToggleConnection = toggleConnection;
            AddNeuron = addNeuron;
            RemoveNeuron = removeNeuron;
            RandomFunction = randomFunction;
            WeightAdjustmentMagnitude = weightAdjustmentMagnitude;
            DefaultActivationFunction = defaultActivationFunction;

            HiddenActivationFunctionPool = hiddenActivationFunctionPool;
            RandomDefaultActivationFunction = randomDefaultActivationFunction;
        }

    }
}
