using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.Hosting;
using System.Threading;

namespace EasyNNFramework.NEAT {
    public static class DebugProgram {

        private static readonly int NetworkCount = 200;
        private static readonly float MutationChance = 0.5f;
        private static readonly int MutationCount = 8;
        private static readonly int MaxGenerations = 1000000;

        //includes all activation functions as mutation possibility
        private static readonly ActivationFunction[] ActivationFunctionPool = (ActivationFunction[])Enum.GetValues(typeof(ActivationFunction));

        //DefaultActivationFunction is not specified because we use RandomDefaultActivationFunction
        private static readonly MutateOptions MOptions = new MutateOptions(0.10f, 0.07f, 0.65f, 0.05f, 0.03f, 0.03f, 0.05f, 0.02f, default, ActivationFunctionPool, true);
        
        //compatabilityThreshold defines how similar 2 networks must be, in order to be the same species
        //maxSpecies is only used when using the frameworks method for evaluating the next generation
        private static readonly SpeciationOptions SOptions = new SpeciationOptions(1, 0.05f, 0.70f, 10, false);

        static void Main(string[] args) {
            
            //restart point
            restart:

            //get input/output neuron templates
            GetTemplates(out List<Neuron> insN, out List<Neuron> outsN);

            //create NEAT object which handles all neural networks
            Neat neat = new Neat(insN.ToArray(), outsN.ToArray(), SOptions);

            //populate NEAT
            for (int i = 0; i < NetworkCount; i++) neat.AddNetwork(i);

            //speciate all networks
            neat.SpeciateAll();
            neat.NetworkCollection[1].Fitness = 1f;

            //using stopwatch to see performance of algorithm
            Stopwatch run = new Stopwatch();
            run.Start();

            int currentGeneration = 1;
            do {
                if (currentGeneration >= MaxGenerations) break;

                //evaluate new species
                var newPop = neat.SpeciesPopulation(NetworkCount, 1);
                Dictionary<int, Network> bestOfSpecie = new Dictionary<int, Network>(newPop.Count);
                foreach (var specie in newPop) {
                    bestOfSpecie.Add(specie.Item1, neat.Species[specie.Item1].AllNetworks.Values.OrderByDescending(o => o.Fitness).First());
                }

                //remove old
                neat.RemoveAllNetworks();

                //create new species
                foreach (var specie in newPop) {
                    for (int i = 0; i < specie.Item2; i++) {
                        int networkID = neat.AddNetwork(neat.NetworkCollection.Count).NetworkID;
                        neat.ChangeNetwork(networkID, bestOfSpecie[specie.Item1]);

                        //mutate network randomly
                        if (neat.Random.NextDouble() <= MutationChance) {
                            int rndCount = neat.Random.Next(1, MutationCount + 1);
                            for (int j = 0; j < rndCount; j++) neat.NetworkCollection[networkID].Mutate(neat, MOptions);
                        }
                    }
                }
                
                neat.SpeciateAll();
                neat.RemoveEmptySpecies();

                TestFitness(neat);
                //RunAndFitnessXOR(neat);

                /*
                //Debug
                //every 50 gens
                if (generations % 10 == 0) {
                    var pop = neat.SpeciesPopulation(NetworkCount, 1).ToDictionary(o => o.Item1, o => o.Item2);
                    Console.WriteLine("Population size: " + neat.NetworkCollection.Count);
                    Console.WriteLine("Generation: " + generations);
                    foreach (var specie in neat.Species) {
                        if(specie.Value.AllNetworks.Count == 0) continue;
                        Network best = specie.Value.AllNetworks.Values.OrderByDescending(o => o.Fitness).First();
                        Console.WriteLine(".......................Specie " + specie.Key + ".....................");
                        Console.WriteLine("Genome: " + String.Join(".", specie.Value.Representative.Connections.Select(o => o.Key).Concat(specie.Value.Representative.RecurrentConnections.Select(o => o.Key)).OrderBy(o => o)));
                        Console.WriteLine("Avg. Fitness: " + specie.Value.AverageFitness(false));
                        Console.WriteLine("Steps since improvement: " + specie.Value.StepsSinceImprovement);
                        Console.WriteLine("Specie Size: " + specie.Value.AllNetworks.Count);
                        Console.WriteLine("New Specie Size: " + (pop.ContainsKey(specie.Key) ? pop[specie.Key].ToString() : "none"));
                        Console.WriteLine("Best Fitness: " + best.Fitness);
                        Console.WriteLine("Best Neuron count: " + best.HiddenNeurons.Length);
                    }
                }*/
                Console.Write("\rCurrent generation: {0} Species Amount: {1}", currentGeneration, neat.Species.Count);
                currentGeneration++;

            } while (true /*neat.NetworkCollection.Max(o => o.Value.Fitness) < 0.98f*/);
            
            //get performance of algorithm
            run.Stop();
            Console.WriteLine("\rGenerations: " + currentGeneration + "  Time: " + run.ElapsedMilliseconds / 1000f + "s \n");
            run.Reset();

            goto restart;

            Console.WriteLine("Species: " + neat.NetworkCollection.OrderByDescending(o => o.Value.Fitness).First().Value.SpeciesID);
            Console.WriteLine(neat.NetworkCollection.OrderByDescending(o => o.Value.Fitness).First().Value.HiddenNeurons.Length);
            Console.WriteLine("Time: " + run.ElapsedMilliseconds / 1000f + "s");

            Console.Read();
        }

        public static float[] ArrayBuffer;
        public static void TestFitness(Neat neat) {
            if(ArrayBuffer == null) ArrayBuffer = new float[neat.InputTemplate.Length];

            foreach (var network in neat.NetworkCollection) {

                //fill inputs with random values
                for (int i = 0; i < ArrayBuffer.Length; i++) {
                    ArrayBuffer[i] = (float)neat.Random.NextDouble() - (float)neat.Random.NextDouble();
                }

                network.Value.InputValues = ArrayBuffer;
                network.Value.CalculateNetwork();

                network.Value.Fitness = network.Value.Neurons.Count;
            }
        }

        public static void RunAndFitnessXOR(Neat neat) {

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
        
        public static float EvaluateXOR(int in1, int in2, int in3, Network network) {

            //set inputs
            network.InputValues = new [] { in1, in2, in3, 1f };

            network.CalculateNetwork();

            //expected value
            int expectedOutput = LogicXOR(LogicXOR(in1, in2), in3);
            
            return 1f - Math.Abs(network.OutputValues[0] - expectedOutput);
        }

        public static int LogicXOR(int in1, int in2) {
            if (in1 != in2) {
                return 1;
            }

            return 0;
        }

        public static void GetTemplates(out List<Neuron> input, out List<Neuron> output) {
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
