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
                networks.Add(new KeyValuePair<NEAT, float>(getStartingNetwork(), 0f));
            }

            //training
            float generations = 1f;
            networks = runXOR(networks, rnd);
            Stopwatch w = new Stopwatch();
            do {
                if (generations >= maxGens) break;

                w.Start();
                //get best performing networks
                networks.RemoveRange(0, networkCount/2);
                for (int i = 0; i < networkCount/2 ; i++) {
                    networks.Add(new KeyValuePair<NEAT, float>(new NEAT(networks[i].Key), networks[i].Value));
                }
                w.Stop();

                Console.WriteLine("Cloning time: " + w.ElapsedMilliseconds);

                w.Reset();

                w.Start();
                for (int i= 0; i < networks.Count-5; i++) {
                    if (rnd.NextDouble() < rndMutation) {

                        int rndNr = rnd.Next(0, 10);
                        for (int j = 0; j < rndNr; j++) {
                            networks[i].Key.Mutate(rnd, 10f, 5f, 10f, 15f, 5f, 5f, 0f, 50f, ActivationFunction.SWISH);
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

                for (int i = 0; i < 10; i++) {
                    Console.WriteLine("Fitness:" + networks[190 + i].Value);
                }

                if (generations % 50 == 0) {
                    Console.WriteLine("Connection count: " + networks.Last().Key.connectionList.Length);
                    Console.WriteLine("Recurrent Connection count: " + networks.Last().Key.recurrentConnectionList.Length);
                    Console.WriteLine("Neuron count: " + networks.Last().Key.hiddenNeurons.Length);
                    Console.WriteLine("Average fitness: " + (networks.Skip(189).Sum(x => x.Value) / 9f) );
                }

            } while (networks.Count(o => o.Value >= 0.98f) < 10);

            Console.Read();
        }

        private static List<KeyValuePair<NEAT, float>> runXOR(List<KeyValuePair<NEAT, float>> nets, Random rnd) {
            List<KeyValuePair<NEAT, float>> calculatedNets = new List<KeyValuePair<NEAT, float>>();

            NEAT neat;
            for (int i = 0; i < nets.Count; i++) {

                neat = nets[i].Key;

                float fitness = 0f;
                fitness += calculateXOR(0, 0, 0, ref neat);
                fitness += calculateXOR(0, 0, 1, ref neat);
                fitness += calculateXOR(0, 1, 0, ref neat);
                fitness += calculateXOR(0, 1, 1, ref neat);
                fitness += calculateXOR(1, 0, 0, ref neat);
                fitness += calculateXOR(1, 0, 1, ref neat);
                fitness += calculateXOR(1, 1, 0, ref neat);
                fitness += calculateXOR(1, 1, 1, ref neat);

                calculatedNets.Add(new KeyValuePair<NEAT, float>(neat, fitness/8f));
            }

            return calculatedNets.OrderBy(o => o.Value).ToList();
        }

        private static float[] outs = new float[1];
        private static float[] ins = new float[4];
        private static float calculateXOR(int in1, int in2, int in3, ref NEAT neat) {

            //set inputs
            ins[0] = in1;
            ins[1] = in2;
            ins[2] = in3;
            ins[3] = 1f;
            
            neat.CalculateNetwork(ins, ref outs);

            //expected value
            int expectedOutput = logicXOR(logicXOR(in1, in2), in3);

            //converging points (amount of correct / amount of input configs)
            return 1f - Math.Abs(outs[0] - expectedOutput);
        }

        private static int logicXOR(int in1, int in2) {
            if (in1 != in2) {
                return 1;
            }

            return 0;
        }

        private static NEAT getStartingNetwork() {
            getLists(out List<Neuron> ins, out List<Neuron> outs);
            return new NEAT(ins.ToArray(), outs.ToArray());
        }

        private static void getLists(out List<Neuron> input, out List<Neuron> output) {
            Neuron in1 = new Neuron( 0, ActivationFunction.IDENTITY, NeuronType.Input);
            Neuron in2 = new Neuron(1, ActivationFunction.IDENTITY, NeuronType.Input);
            Neuron in3 = new Neuron(2, ActivationFunction.IDENTITY, NeuronType.Input);
            Neuron bias = new Neuron(3, ActivationFunction.IDENTITY, NeuronType.Input);
            Neuron out1 = new Neuron(4, ActivationFunction.TANH, NeuronType.Action);

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
