using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyNNFramework.NEAT {

    [Serializable]
    public struct LayerStructure {
        public List<int>[] LayerArray;
        private Dictionary<int, int> _neuronLayerDict;

        private int _highestLayer;

        public LayerStructure(in Network network) {
            LayerArray = Array.Empty<List<int>>();
            _neuronLayerDict = new Dictionary<int, int>(network.Neurons.Count); //key = neuron id, value = layer
            _highestLayer = -1;

            // TODO directly add neurons to layerArray without using neuronLayerDict as a buffer, maybe adding layer variable in neuron class?

            //add input neurons
            _highestLayer = 1;
            foreach (var input in network.InputNeurons) {
                _neuronLayerDict.Add(input.ID, 1);
            }

            //calculate hidden neuron layers by back propagating connections
            foreach (Neuron n in network.HiddenNeurons) {
                GetLayer(n.ID, network);    //automatically adds neurons to neuronLayerDict
            }

            //action neurons must be manually added because they are not seen in the back propagation process
            _highestLayer++;
            foreach (var neuron in network.ActionNeurons) {
                _neuronLayerDict.Add(neuron.ID, _highestLayer);
            }

            //init layer array
            //make sure to call this after GetLayer because that's where _highestLayer is defined
            LayerArray = new List<int>[_highestLayer];
            for (int i = 0; i < _highestLayer; i++) LayerArray[i] = new List<int>();

            //populate layer array
            //input neurons
            for (int i = 0; i < network.InputNeurons.Length; i++) {
                LayerArray[0].Add(network.InputNeurons[i].ID);
            }
            //hidden neurons
            for (int i = 0; i < network.HiddenNeurons.Length; i++) {
                int layerIndex = _neuronLayerDict[network.HiddenNeurons[i].ID] - 1;
                LayerArray[layerIndex].Add(network.HiddenNeurons[i].ID);
            }
            //action neurons
            for (int i = 0; i < network.ActionNeurons.Length; i++) {
                LayerArray[_highestLayer - 1].Add(network.ActionNeurons[i].ID);
            }
        }

        //automatically adds to internal neuron layer dict
        // TODO rewrite function to also handle input neurons which might allow for more performant ways of creating the layer structure
        public int GetLayer(int neuronID, in Network network) {
            if (_neuronLayerDict.TryGetValue(neuronID, out int v)) return v;


            int highestSourceNeuronLayer = 1;
            for (int i = 0; i < network.Neurons[neuronID].IncommingConnections.Count; i++) {
                Connection con = network.Connections[network.Neurons[neuronID].IncommingConnections[i]];
                int l = GetLayer(con.SourceID, network);
                if (l > highestSourceNeuronLayer) highestSourceNeuronLayer = l;
            }

            if(!_neuronLayerDict.ContainsKey(neuronID)) _neuronLayerDict.Add(neuronID, highestSourceNeuronLayer + 1);

            if (highestSourceNeuronLayer + 1 > _highestLayer) _highestLayer = highestSourceNeuronLayer + 1;
            return highestSourceNeuronLayer + 1;
        }
    }
}
