using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using EasyNNFramework.NEAT;

namespace EasyNNFramework.NEAT {
    public static class DebugProgram {

        private const int networkCount = 200;
        private const float mutationChance = 0.25f;
        private const int maxGens = 100000;

        static void Main(string[] args) {

            GetTemplates(out List<Neuron> insN, out List<Neuron> outsN);
            Neat neat = new Neat(insN.ToArray(), outsN.ToArray(), new SpeciationOptions(1, 0.05f, 0.70f, 10, false, false, 15));
            neat.AddNetwork(networkCount);

            //training
            float generations = 1f;
            Stopwatch w = new Stopwatch();
            Stopwatch run = new Stopwatch();
            run.Start();
            do {
                if (generations >= maxGens) break;

                //repopulate
                w.Start();
                neat.Repopulate(networkCount, false, mutationChance, 4, new MutateOptions(0.05f, 0.0f, 0.05f, 0.80f, 0.05f, 0.05f, 0f, 0f, ActivationFunction.TANH));

                w.Stop();
                Console.WriteLine("Repopulating time: " + w.ElapsedMilliseconds);
                w.Reset();

                //calculate
                w.Start();
                RunAndFitnessXOR(neat);

                w.Stop();
                Console.WriteLine("Calculating time: " + w.ElapsedMilliseconds);
                w.Reset();

                //Debug
                //every 50 gens
                if (generations % 10 == 0) {
                    var pop = neat.SpeciesPopulation(networkCount).ToDictionary(o => o.Item1, o => o.Item2);
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
                }

                generations++;

            } while (neat.NetworkCollection.OrderByDescending(o => o.Value.Fitness).First().Value.Fitness < 0.98f);
            run.Stop();

            Console.WriteLine("Species: " + neat.NetworkCollection.OrderByDescending(o => o.Value.Fitness).First().Value.SpeciesID);
            Console.WriteLine(neat.NetworkCollection.OrderByDescending(o => o.Value.Fitness).First().Value.HiddenNeurons.Length);
            Console.WriteLine("Time: " + run.ElapsedMilliseconds / 1000f + "s");

            Console.Read();
        }

        public static void RunAndFitnessXOR(Neat neat) {

            foreach (var network in neat.NetworkCollection) {
                
                network.Value.ResetFitness();
                float fitness = 0;
                fitness += EvaluateXOR(0, 0, 0, network.Value);
                fitness += EvaluateXOR(0, 0, 1, network.Value);
                fitness += EvaluateXOR(0, 1, 0, network.Value);
                fitness += EvaluateXOR(0, 1, 1, network.Value);
                fitness += EvaluateXOR(1, 0, 0, network.Value);
                fitness += EvaluateXOR(1, 0, 1, network.Value);
                fitness += EvaluateXOR(1, 1, 0, network.Value);
                fitness += EvaluateXOR(1, 1, 1, network.Value);
                network.Value.AddFitness(fitness / 8f);
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
