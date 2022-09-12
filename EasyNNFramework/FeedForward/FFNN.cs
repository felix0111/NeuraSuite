using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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


        public FFNN(int layers, List<Neuron> _inputNeurons, int hiddenNeuronsPerLayer, List<Neuron> _actionNeurons) {
            layerCount = layers;

            inputNeurons = _inputNeurons;
            actionNeurons = _actionNeurons;
            hiddenNeurons = new List<List<Neuron>>();

            for (int i = 0; i < layerCount; i++) {
                hiddenNeurons.Add(new List<Neuron>());
                for (int j = 0; j < hiddenNeuronsPerLayer; j++) {
                    hiddenNeurons[i].Add(new Neuron(i + "hidden" + j, NeuronType.Hidden));
                }
            }

            weightHandler = new WeightHandler(this);

        }

    }

}


