using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using EasyNNFramework.NEAT;

namespace EasyNNFramework.NEAT {
    public static class DebugProgram {

        private const int networkCount = 200;
        private const float rndMutation = 1f;
        private const int maxGens = 10000;

        static void Main(string[] args) {

            Random rnd = new Random();

            List<KeyValuePair<NEAT, float>> networks = new List<KeyValuePair<NEAT, float>>();

            //populate dict
            for (int i = 0; i < networkCount; i++) {
                networks.Add(new KeyValuePair<NEAT, float>(getStartingNetwork(5, rnd), 0f));
            }

            //training
            float generations = 1f;
            networks = runXOR(networks, rnd);
            Stopwatch w = new Stopwatch();
            networks.First().Key.layerManager.inputLayer.neurons.First().Value.value = 1f;
            do {
                if (generations >= maxGens) break;

                w.Start();
                //get best performing networks
                networks.RemoveRange(0, networkCount/2);
                for (int i = 0; i < networkCount/2 ; i++) {
                    //networks.Add(new KeyValuePair<NEAT, float>(networks[i].Key.DeepClone(), networks[i].Value));
                    networks.Add(new KeyValuePair<NEAT, float>(new NEAT(networks[i].Key), networks[i].Value));
                }
                w.Stop();
                Console.WriteLine("Cloning time: " + w.ElapsedMilliseconds);
                w.Reset();

                w.Start();
                foreach (KeyValuePair<NEAT, float> pair in networks.Take(networks.Count-10)) {
                    if (rnd.NextDouble() < rndMutation) {
                        for (int j = 0; j < rnd.Next(1, 20); j++) {
                            pair.Key.Mutate(rnd, 25f, 0f, 5f, 3f, 7f, 0f, 0f, 60f, ActivationFunction.SWISH);
                        }
                    }
                }
                w.Stop();
                Console.WriteLine("Mutating time: " + w.ElapsedMilliseconds);
                w.Reset();

                w.Start();
                networks = runXOR(networks, rnd);
                w.Stop();
                Console.WriteLine("Calculating time: " + w.ElapsedMilliseconds);
                generations++;

                if (generations % 50 == 0) {
                    Console.WriteLine("Connection count: " + networks.Last().Key.connectionList.Count);
                    Console.WriteLine("Neuron count: " + (networks.Last().Key.layerManager.allLayers.Sum(o => o.neurons.Count) - 3));
                    Console.WriteLine("Average fitness: " + (networks.Skip(190).Sum(x => x.Value) / 9f) );
                }

            } while (networks.Count(o => o.Value >= 0.98f) < 10);

            Console.Read();
        }

        private static List<KeyValuePair<NEAT, float>> runXOR(List<KeyValuePair<NEAT, float>> nets, Random rnd) {
            List<KeyValuePair<NEAT, float>> calculatedNets = new List<KeyValuePair<NEAT, float>>();

            for (int i = 0; i < nets.Count; i++) {

                NEAT neat = nets[i].Key;

                float fitness = 0f;
                fitness += calculateXOR(0, 0, 0, neat);
                fitness += calculateXOR(0, 0, 1, neat);
                fitness += calculateXOR(0, 1, 0, neat);
                fitness += calculateXOR(0, 1, 1, neat);
                fitness += calculateXOR(1, 0, 0, neat);
                fitness += calculateXOR(1, 0, 1,neat);
                fitness += calculateXOR(1, 1, 0,neat);
                fitness += calculateXOR(1, 1, 1,neat);

                calculatedNets.Add(new KeyValuePair<NEAT, float>(neat, fitness/8f));
            }

            return calculatedNets.OrderBy(o => o.Value).ToList();
        }

        private static float calculateXOR(int in1, int in2, int in3, NEAT neat) {

            //set inputs
            neat.layerManager.inputLayer.neurons[0].value = in1;
            neat.layerManager.inputLayer.neurons[1].value = in2;
            neat.layerManager.inputLayer.neurons[2].value = in3;
            
            neat.calculateNetwork();

            //expected value
            int expectedOutput = logicXOR(logicXOR(in1, in2), in3);

            //out
            float output = neat.layerManager.actionLayer.neurons.Values.ElementAt(0).value;
            
            //converging points (amount of correct / amount of input configs)
            return 1f - Math.Abs(output - expectedOutput);
        }

        private static int logicXOR(int in1, int in2) {
            if (in1 != in2) {
                return 1;
            }

            return 0;
        }

        private static NEAT getStartingNetwork(int startingMutations, Random rnd) {
            getDicts(out List<Neuron> ins, out List<Neuron> outs);
            return new NEAT(ins, outs);
        }

        private static void getDicts(out List<Neuron> input, out List<Neuron> output) {
            Neuron in1 = new Neuron( -1, ActivationFunction.IDENTITY);
            Neuron in2 = new Neuron(-1, ActivationFunction.IDENTITY);
            Neuron in3 = new Neuron(-1, ActivationFunction.IDENTITY);
            Neuron bias = new Neuron(-1, ActivationFunction.IDENTITY);
            Neuron out1 = new Neuron(-1, ActivationFunction.SIGMOID);

            input = new List<Neuron>();
            output = new List<Neuron>();

            bias.value = 1f;

            input.Add(in1);
            input.Add(in2);
            input.Add(in3);
            input.Add(bias);
            output.Add(out1);
        }
    }
}
