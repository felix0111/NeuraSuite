using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace EasyNNFramework {
    public static class DebugProgram {
        private static Neuron test1, test2;
        private static Neuron testOut1, testOut2;


        static void Main(string[] args) {
            test1 = new Neuron("Test1", NeuronType.Input, default);
            test2 = new Neuron("Test2", NeuronType.Input, default);
            testOut1 = new Neuron("TestOut1", NeuronType.Action, ActivationFunction.TANH);
            testOut2 = new Neuron("TestOut2", NeuronType.Action, ActivationFunction.TANH);

            List<Neuron> inputs = new List<Neuron>();
            List<Neuron> action = new List<Neuron>();

            inputs.Add(test1);
            inputs.Add(test2);
            inputs.Add(new Neuron("Test3", NeuronType.Input, default));
            inputs.Add(new Neuron("Test4", NeuronType.Input, default));
            inputs.Add(new Neuron("Test5", NeuronType.Input, default));
            inputs.Add(new Neuron("Test6", NeuronType.Input, default));
            inputs.Add(new Neuron("Test7", NeuronType.Input, default));
            inputs.Add(new Neuron("Test8", NeuronType.Input, default));
            inputs.Add(new Neuron("Test9", NeuronType.Input, default));
            inputs.Add(new Neuron("Test10", NeuronType.Input, default));
            inputs.Add(new Neuron("Test11", NeuronType.Input, default));
            inputs.Add(new Neuron("Test12", NeuronType.Input, default));
            action.Add(testOut1);
            action.Add(testOut2);
            action.Add(new Neuron("TestOut3", NeuronType.Action, ActivationFunction.TANH));
            action.Add(new Neuron("TestOut4", NeuronType.Action, ActivationFunction.TANH));
            action.Add(new Neuron("TestOut5", NeuronType.Action, ActivationFunction.TANH));
            action.Add(new Neuron("TestOut6", NeuronType.Action, ActivationFunction.TANH));
            action.Add(new Neuron("TestOut7", NeuronType.Action, ActivationFunction.TANH));
            
            NEAT neatTest = new NEAT(inputs, action);
            Stopwatch watch = new Stopwatch();
            

            test1.value = 2f;
            test2.value = 1f;
            
            watch.Start();
            for (int i = 0; i < 50; i++) {
                neatTest.Mutate();
            }
            watch.Stop();
            Console.WriteLine("Mutate took: " + watch.ElapsedMilliseconds);

            watch.Restart();
            for (int i = 0; i < 50; i++) {
                neatTest.calculateNetwork();
            }
            watch.Stop();
            Console.WriteLine("Calculate took: " + watch.ElapsedMilliseconds);

            Console.WriteLine("TestOut1: " + testOut1.value);
            Console.WriteLine("TestOut2: " + testOut2.value);

            Console.Read();
        }
    }
}
