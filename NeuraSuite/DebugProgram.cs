using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using NeuraSuite.NEAT;

namespace NeuraSuite {
    public static class DebugProgram {

        private static readonly int NetworkCount = 200;
        private static readonly float MutationChance = 0.5f;
        private static readonly int MutationCount = 4;
        private static readonly int MaxGenerations = 10000;

        //includes all activation functions as mutation possibility
        private static readonly ActivationFunction[] ActivationFunctionPool = (ActivationFunction[])Enum.GetValues(typeof(ActivationFunction));

        //DefaultActivationFunction is not specified because we use RandomDefaultActivationFunction
        private static readonly MutateOptions MOptions = new MutateOptions(0.10f, 0.07f, 0.65f, 0.05f, 0.03f, 0.03f, 0.05f, 0.02f, default, ActivationFunctionPool, true);
        
        private static readonly SpeciationOptions SOptions = new SpeciationOptions(1, 0f, 0.6f, 20, false);

        static void Main(string[] args) {
            
            //restart point
            restart:

            //create input/output neuron templates
            GetXORTemplates(out List<Neuron> insN, out List<Neuron> outsN);

            //create NEAT object which handles all neural networks
            Neat neat = new Neat(insN.ToArray(), outsN.ToArray(), SOptions);

            //populate NEAT
            for (int i = 0; i < NetworkCount; i++) neat.AddNetwork(i);

            //speciate all networks, this should put all networks in the same species
            neat.SpeciateAll();

            //needed because else neat.CreatePopulation would return an empty list in the first iteration (fitness of every network is 0)
            neat.NetworkCollection[1].Fitness = 1f;

            //using stopwatch to see performance of algorithm
            Stopwatch run = new Stopwatch();
            run.Start();

            int currentGeneration = 1;
            Network bestNetwork;
            do {
                if (currentGeneration >= MaxGenerations) break;

                //create a new population, basically tells how many network each species will get
                var newPop = neat.CreatePopulation(NetworkCount, 1);

                //TODO might have to check if newPop is not empty

                //take the best network of each species
                Dictionary<int, Network> bestOfSpecies = new Dictionary<int, Network>(newPop.Count);
                foreach (var specie in newPop) {
                    bestOfSpecies.Add(specie.Item1, neat.Species[specie.Item1].AllNetworks.Values.OrderByDescending(o => o.Fitness).First());
                }

                //remove all old networks
                neat.RemoveAllNetworks();

                //create the new population based of newPop, species.Item1 => speciesID; species.Item2 => network amount
                foreach (var species in newPop) {
                    for (int i = 0; i < species.Item2; i++) {
                        var network = neat.AddNetwork(neat.NetworkCollection.Count, bestOfSpecies[species.Item1]);

                        //mutate network randomly
                        if (neat.Random.NextDouble() <= MutationChance) {
                            int rndCount = neat.Random.Next(1, MutationCount + 1);
                            for (int j = 0; j < rndCount; j++) neat.NetworkCollection[network.NetworkID].Mutate(neat, MOptions);
                        }
                    }
                }


                //neat.AdjustCompatabilityFactor(0.01f, 60);
                
                //speciate every network because mutation might throw a network out of its species
                neat.SpeciateAll();
                neat.RemoveEmptySpecies();


                TestXOR(neat);


                //show some data
                bestNetwork = neat.NetworkCollection.MaxBy(o => o.Value.Fitness).Value;
                Console.Write("\rCurrent generation: {0:D3} Species amount: {1:D3} Comp.threshold: {2:F2} Best accuracy: {3:F1}% accuracy", currentGeneration, neat.Species.Count, neat.SpeciationOptions.CompatabilityThreshold, bestNetwork.Fitness * 100f);
                currentGeneration++;

            } while (bestNetwork.Fitness <= 0.99f);
            
            //show performance of population, overall performance might be affected by Console.Write(...) in while loop
            run.Stop();
            Console.WriteLine("\nTotal amount of generations: {0} Time elapsed: {1}s", currentGeneration, run.ElapsedMilliseconds / 1000f);
            run.Reset();

            Console.WriteLine("Enter 'exit' to close.");
            if (Console.ReadLine() == "exit") return;
            goto restart;
        }

        /// <summary>
        /// Rewards networks with many neurons. Just for testing/debugging performance because it does not really make much sense for anything other.
        /// </summary>
        public static void PerformanceTest(Neat neat) {
            foreach (var network in neat.NetworkCollection) {

                //fill inputs with random values
                for (int i = 0; i < network.Value.InputValues.Length; i++) {
                    network.Value.InputValues[i] = (float)neat.Random.NextDouble() - (float)neat.Random.NextDouble();
                }

                network.Value.CalculateNetwork();
                network.Value.Fitness = network.Value.Neurons.Count;
            }
        }

        /// <summary>
        /// Tests how good each neural network solves a 3 input XOR logic gate. Very simple neural network test.
        /// </summary>
        public static void TestXOR(Neat neat) {

            foreach (var network in neat.NetworkCollection) {
                
                float fitness = 0;
                fitness += EvaluateXOR(0, 0, 0, network.Value);
                fitness += EvaluateXOR(0, 0, 1, network.Value);
                fitness += EvaluateXOR(0, 1, 0, network.Value);
                fitness += EvaluateXOR(0, 1, 1, network.Value);
                fitness += EvaluateXOR(1, 0, 0, network.Value);
                fitness += EvaluateXOR(1, 0, 1, network.Value);
                fitness += EvaluateXOR(1, 1, 0, network.Value);
                fitness += EvaluateXOR(1, 1, 1, network.Value);
                network.Value.Fitness = fitness / 8f;
            }
        }
        
        /// <summary>
        /// Returns the difference between the expected XOR result and the network result, a value between 0 and 1
        /// </summary>
        public static float EvaluateXOR(int in1, int in2, int in3, Network network) {

            //set inputs, 4th input is bias
            network.InputValues = new [] { in1, in2, in3, 1f };

            network.CalculateStep();

            //expected value
            int expectedOutput = LogicXOR(LogicXOR(in1, in2), in3);

            //debug
            if (1f - Math.Abs(network.OutputValues[0] - expectedOutput) < 0f || 1f - Math.Abs(network.OutputValues[0] - expectedOutput) > 1f) throw new Exception("");

            //difference between expected value and network output
            return 1f - Math.Abs(network.OutputValues[0] - expectedOutput);
        }

        /// <summary>
        /// Get the logic XOR result.
        /// </summary>
        public static int LogicXOR(int in1, int in2) {
            if (in1 != in2) {
                return 1;
            }

            return 0;
        }

        /// <summary>
        /// Creates input and output neuron templates matching the XOR test.
        /// </summary>
        public static void GetXORTemplates(out List<Neuron> input, out List<Neuron> output) {
            Neuron in1 = new Neuron( 0, ActivationFunction.IDENTITY, NeuronType.Input);
            Neuron in2 = new Neuron(1, ActivationFunction.IDENTITY, NeuronType.Input);
            Neuron in3 = new Neuron(2, ActivationFunction.IDENTITY, NeuronType.Input);
            Neuron bias = new Neuron(3, ActivationFunction.IDENTITY, NeuronType.Bias);
            Neuron out1 = new Neuron(4, ActivationFunction.SIGMOID, NeuronType.Action);

            input = new List<Neuron>();
            output = new List<Neuron>();

            input.Add(in1);
            input.Add(in2);
            input.Add(in3);
            input.Add(bias);
            output.Add(out1);
        }
    }
}
