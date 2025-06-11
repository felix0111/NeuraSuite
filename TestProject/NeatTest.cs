using NeuraSuite.Neat;
using NeuraSuite.Neat.Core;

namespace TestProject {
    public static class NeatTest {

        public static void RunTest() {

            MutationSettings mo = new (0.05D, 0.03D, 0.8D, 2D);
            SpeciationSettings ss = new (1D, 1D, 0.4D, 3D);
            NeatSettings no = new (150, 0.10D, 0.75D, 0.25D, 1500, 1500);

            NeatManager nm = new NeatManager(GetPresetGenome(), no, mo, ss);

            double bestFitness = 0D;
            int counter = 0;
            while (bestFitness < 3.9D && counter < 1000) {
                //tests and calculates fitness for the xor problem on current population
                var phenos = nm.CreatePhenotypes();
                TestXor(phenos);

                bestFitness = Math.Round(Math.Sqrt(nm.EntirePopulation.Max(o => o.Fitness)), 5);

                //log
                Console.Clear();
                Console.WriteLine("Generation {0}", counter);
                Console.WriteLine("Best Fitness of population: {0}", bestFitness);
                int i = 1;
                foreach (var species in nm.Species.Where(o => o.Members.Count != 0)) {
                    Console.WriteLine("--Species {0} with {1} members--", i++, species.Members.Count);
                    Console.WriteLine("    Avg. Fitness: {0}", Math.Round(Math.Sqrt(species.Members.Average(o => o.Fitness)), 5));
                    var champion = phenos.MaxBy(o => o.Genome.Fitness);
                    Console.WriteLine("    Champion: Fitness: {0} Nodes: {1} Connections: {2}", champion.Genome.Fitness, champion.Genome.Nodes.Values.Count(o => champion.IsNodeUsed(o.Id)), champion.Genome.Connections.Count(o => o.Value.Enabled));
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
                network.Genome.Fitness = Math.Round(Math.Pow(4D - sum, 2), 4);
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