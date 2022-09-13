using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using EasyNNFramework.FeedForward;

namespace EasyNNFramework {

    [Serializable]
    internal class FFNN {

        public int layerCount;

        public List<Neuron> inputNeurons, actionNeurons;
        public List<List<Neuron>> hiddenNeurons;
        private WeightHandler weightHandler;


        public FFNN(int layers, List<Neuron> _inputNeurons, int hiddenNeuronsPerLayer, ActivationFunction hiddenActicationFunction, List<Neuron> _actionNeurons) {
            layerCount = layers;

            inputNeurons = _inputNeurons;
            actionNeurons = _actionNeurons;
            hiddenNeurons = new List<List<Neuron>>();

            for (int i = 0; i < layerCount; i++) {
                hiddenNeurons.Add(new List<Neuron>());
                for (int j = 0; j < hiddenNeuronsPerLayer; j++) {
                    hiddenNeurons[i].Add(new Neuron(i + "hidden" + j, NeuronType.Hidden, hiddenActicationFunction));
                }
            }

            weightHandler = new WeightHandler(this);

        }

        public void setInputNeuronValue(Neuron input, float value) {
            int index = inputNeurons.IndexOf(input);

            if (index != -1) {
                inputNeurons[index].value = value;

            } else {
                throw new KeyNotFoundException("Cannot find input neuron: " + input.name);
            }
        }

        public void calculateNeuronValueWithPrevLayer(Neuron targetNeuron, int prevLayer) {
            if (targetNeuron.type == NeuronType.Action) {
                float sum = 0;
                List<Neuron> lastHiddenLayer = hiddenNeurons.Last();

                foreach (Neuron neuron in lastHiddenLayer) {
                    sum += neuron.value * weightHandler.getWeight(neuron, targetNeuron);
                }

                targetNeuron.value = Neuron.getFunctionValue(targetNeuron.function, sum);
            } else if (targetNeuron.type == NeuronType.Hidden && prevLayer == 0) {
                float sum = 0;
                foreach (Neuron inputNeuron in inputNeurons) {
                    sum += inputNeuron.value * weightHandler.getWeight(inputNeuron, targetNeuron);
                }

                targetNeuron.value = Neuron.getFunctionValue(targetNeuron.function, sum);
            } else if (targetNeuron.type == NeuronType.Hidden) {
                List<Neuron> prevHiddenLayer = hiddenNeurons[prevLayer - 1];
                float sum = 0;
                foreach (Neuron hiddenNeuron in prevHiddenLayer) {
                    sum += hiddenNeuron.value * weightHandler.getWeight(hiddenNeuron, targetNeuron);
                }

                targetNeuron.value = Neuron.getFunctionValue(targetNeuron.function, sum);
            } else {
                throw new NullReferenceException("Previous layer {" + prevLayer +
                                                 "} not existing when calculating values.");
            }
        
        }

        public float getNeuronValue(Neuron targetNeuron) {
            return targetNeuron.value;
        }

        public void setWeightBetweenNeurons(Neuron start, Neuron end, float weight) {
            weightHandler.addWeight(start, end, weight);
        }

    }

}


