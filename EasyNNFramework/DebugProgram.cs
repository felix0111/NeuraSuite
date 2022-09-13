using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyNNFramework {
    internal static class DebugProgram {
        private static Neuron test1, test2;
        private static Neuron testOut1, testOut2;


        static void Main(string[] args) {
            test1 = new Neuron("Test1", NeuronType.Input, default);
            test2 = new Neuron("Test2", NeuronType.Input, default);
            testOut1 = new Neuron("TestOut1", NeuronType.Action, ActivationFunction.GELU);
            testOut2 = new Neuron("TestOut2", NeuronType.Action, ActivationFunction.TANH);

            List<Neuron> inputs = new List<Neuron>();
            List<Neuron> action = new List<Neuron>();

            inputs.Add(test1);
            inputs.Add(test2);
            action.Add(testOut1);
            action.Add(testOut2);
            
            NEAT neatTest = new NEAT(inputs, action);

            test1.value = 2f;
            test2.value = 1f;

            neatTest.Mutate();
            neatTest.Mutate();
            neatTest.Mutate();
            neatTest.Mutate();
            neatTest.Mutate();
            neatTest.Mutate();
            neatTest.Mutate();
            neatTest.Mutate();
            neatTest.calculateNetwork();

            Console.WriteLine("TestOut1: " + testOut1.value);
            Console.WriteLine("TestOut2: " + testOut2.value);

            Console.Read();
        }
    }
}
