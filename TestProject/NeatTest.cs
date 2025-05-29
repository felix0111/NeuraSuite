using NeuraSuite.Neat.Core;
using NeuraSuite.Neat.Utility;

namespace TestProject {
    public class NeatTest {

        public static void RunTest() {

            InnovationManager im = new ();
            MutationOptions mo = new (0.15D, 0.02D, 0.03D, 0.8D, 0.1D);

            //create default nodes
            var list = new List<NodeGene>();
            list.Add(new (im.NewNodeId, NodeType.Input));
            list.Add(new (im.NewNodeId, NodeType.Input));
            list.Add(new (im.NewNodeId, NodeType.Input));
            list.Add(new (im.NewNodeId, NodeType.Output));

            //create genome
            Genome g = new Genome(list);

            //mutate genome
            for (int i = 0; i < 10; i++) {
                g.Mutate(im, mo);
            }

            //create a network from the genome
            Network n = new Network(g);
            
            //set inputs of network
            n.SetValue(0, 1D);
            n.SetValue(1, 1D);
            n.SetValue(2, 1D);
            
            //evaluate the network
            n.Evaluate(1);

            foreach (var node in g.Nodes.Keys) {
                Console.WriteLine(n.GetValue(node));
            }

            Console.ReadLine();
        }

    }
}