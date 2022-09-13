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

            FFNN test = new FFNN(1, inputs, 2, ActivationFunction.TANH, action);

            foreach (Neuron inputNeuron in inputs) {
                foreach (Neuron firstHiddenNeuron in test.hiddenNeurons[0]) {
                    test.setWeightBetweenNeurons(inputNeuron, firstHiddenNeuron, 1);
                }
            }

            for (int i = 0; i < test.layerCount; i++) {
                foreach (Neuron neuronStart in test.hiddenNeurons[i]) {
                    //if at last layer
                    if (i == test.layerCount - 1) {
                        foreach (Neuron actionNeuron in action) {
                            test.setWeightBetweenNeurons(neuronStart, actionNeuron, 1);
                        }
                    }
                    else {
                        foreach (Neuron neuronEnd in test.hiddenNeurons[i+1]) {
                            test.setWeightBetweenNeurons(neuronStart, neuronEnd, 1);
                        }
                    }
                }
            }

            for (int i = 0; i < test.layerCount+2; i++) {

                //if first layer
                if (i == 0) {
                    test.setInputNeuronValue(test1, 2);
                    test.setInputNeuronValue(test2, 2);
                }

                //if not last layer
                else if (i != test.layerCount + 1) {
                    foreach (Neuron hiddenNeuron in test.hiddenNeurons[i-1]) {
                        test.calculateNeuronValueWithPrevLayer(hiddenNeuron, i-1);
                    }
                }

                //if action/last layer
                else {
                    foreach (Neuron actionNeuron in test.actionNeurons) {
                        test.calculateNeuronValueWithPrevLayer(actionNeuron, i-1);
                    }
                }
            }

            Console.WriteLine("TestOut1: " + test.getNeuronValue(testOut1));
            Console.WriteLine("TestOut2: " + test.getNeuronValue(testOut2));

            Console.Read();
        }
    }
}
