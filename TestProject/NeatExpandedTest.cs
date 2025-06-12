using NeuraSuite.NeatExpanded;
using System.Diagnostics;

namespace TestProject
{
    public class NeatExpandedTest {
        private static readonly int NetworkCount = 200;
        private static readonly int MaxGenerations = 10000;

        //includes all activation functions as mutation possibility
        private static readonly ActivationFunction[] ActivationFunctionPool = (ActivationFunction[])Enum.GetValues(typeof(ActivationFunction));

        //DefaultActivationFunction is not specified because we use RandomDefaultActivationFunction
        private static readonly MutateOptions MOptions = new MutateOptions(0.10f, 0.07f, 0.7f, 0.01f, 0.03f, 0.03f, 0.05f, default, ActivationFunctionPool, true);

        private static readonly SpeciationOptions SOptions = new SpeciationOptions(1, 0f, 1f, true);

        public static void RunTest() {

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

            //using stopwatch to see performance of algorithm
            Stopwatch run = new Stopwatch();
            run.Start();

            int currentGeneration = 1;
            Network bestNetwork;
            do {
                TestXOR(neat);

                neat.CompleteGeneration(NetworkCount, 0D, 0.75D, MOptions);
                neat.RemoveEmptySpecies();

                //show some data
                bestNetwork = neat.NetworkCollection.MaxBy(o => o.Value.Fitness).Value;
                Console.Write("\rCurrent generation: {0:D3} Species amount: {1:D3} Comp.threshold: {2:F2} Best accuracy: {3:F1}% accuracy", currentGeneration, neat.Species.Count, neat.SpeciationOptions.CompatabilityThreshold, bestNetwork.Fitness * 100f);
                currentGeneration++;

            } while (bestNetwork.Fitness <= 0.99f && currentGeneration < MaxGenerations);

            //show performance of population, overall performance might be affected by Console.Write(...) in while loop
            run.Stop();
            Console.WriteLine("\nTotal amount of generations: {0} Time elapsed: {1}s Generations Per Second: {2}", currentGeneration, run.ElapsedMilliseconds / 1000f, currentGeneration / (run.ElapsedMilliseconds / 1000f));
            run.Reset();

            Console.WriteLine("Enter 'exit' to stop test.");
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
            network.InputValues = new[] { in1, in2, in3, 1f };

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
            Neuron in1 = new Neuron(0, ActivationFunction.IDENTITY, NeuronType.Input);
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
