using NeuraSuite.Neat;
using NeuraSuite.Neat.Core;
using NeuraSuite.Neat.Utility;

namespace TestProject {
    public static class NeatTest {

        public static void RunTest() {

            MutationOptions mo = new (0.05D, 0.03D, 0.8D, 0.1D);
            NeatOptions no = new (150, 0.05D, 0.75D, 0.25D, 20, 0.8D);

            NeatManager nm = new NeatManager(GetPresetGenome(), no, mo);

            int counter = 0;
            while (counter < 10000) {
                //tests and calculates fitness for the xor problem on current population
                var phenos = nm.CreatePhenotypes();
                TestXor(phenos);

                //log
                Console.Clear();
                Console.WriteLine("Generation {0}", counter);
                Console.WriteLine("Best Fitness of population: {0}", Math.Round(nm.EntirePopulation.Max(o => o.Fitness), 5));
                int i = 1;
                foreach (var species in nm.Species.Where(o => o.Members.Count != 0)) {
                    Console.WriteLine("--Species {0} with {1} members--", i++, species.Members.Count);
                    Console.WriteLine("    Avg. Fitness: {0}", Math.Round(species.Members.Average(o => o.Fitness), 5));
                    var champion = species.Members.MaxBy(o => o.Fitness);
                    Console.WriteLine("    Champion: Fitness: {0} Nodes: {1} Connections: {2}", champion.Fitness, champion.Nodes.Count, champion.Connections.Count);
                }

                //create offspring, mutate
                nm.CompleteGeneration();

                counter++;
            }

            Console.ReadLine();
        }

        public static void TestXor(List<Network> phenotypes) {
            foreach (var network in phenotypes) {
                double sum = GetXorError(network, false, false);
                sum += GetXorError(network, false, true);
                sum += GetXorError(network, true, false);
                sum += GetXorError(network, true, true);
                network.Genome.Fitness = 1D - sum/4D;
            }
        }

        public static double GetXorError(Network network, bool in1, bool in2) {
            network.Reset();

            network.SetValue(0, 1); //bias
            network.SetValue(1, in1 ? 1 : 0); //input 1
            network.SetValue(2, in2 ? 1 : 0); //input 2
            network.Evaluate(10);
            double output = network.GetValue(3);
            double expected = in1 != in2 ? 1 : 0;
            return Math.Abs(output - expected);
        }

        public static Genome GetPresetGenome() {
            //create default nodes
            var nodes = new List<NodeGene>();
            nodes.Add(new(0, NodeType.Input)); //bias
            nodes.Add(new(1, NodeType.Input)); //input 1
            nodes.Add(new(2, NodeType.Input)); //input 2
            nodes.Add(new(3, NodeType.Output)); //output

            //connect all inputs to the output nodes
            var connections = new List<ConnectionGene>();
            connections.Add(new ConnectionGene(0, 0, 3));
            connections.Add(new ConnectionGene(1, 1, 3));
            connections.Add(new ConnectionGene(2, 2, 3));

            return new Genome(nodes, connections);
        }

    }
}