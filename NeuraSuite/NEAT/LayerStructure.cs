using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuraSuite.NEAT {

    [Serializable]
    public struct LayerStructure {
        public List<List<int>> LayerArray;
        private Dictionary<int, int> _neuronLayerDict;

        public LayerStructure(in Network network) {
            LayerArray = new List<List<int>>();
            _neuronLayerDict = new Dictionary<int, int>(network.Neurons.Count); //key = neuron id, value = layer

            //add input neurons
            LayerArray.Add(new List<int>(network.InputNeurons.Length));
            foreach (var input in network.InputNeurons) {
                _neuronLayerDict.Add(input.ID, 1);
                LayerArray[0].Add(input.ID);
            }

            //calculate hidden neuron layers by back propagating connections
            foreach (Neuron n in network.HiddenNeurons) {
                int l = GetLayer(n.ID, network);    //automatically adds neurons to neuronLayerDict

                while (LayerArray.Count < l) LayerArray.Add(new List<int>());
                LayerArray[l-1].Add(n.ID);
            }

            //action neurons must be manually added because they are not seen in the back propagation process
            LayerArray.Add(new List<int>(network.ActionNeurons.Length));
            foreach (var neuron in network.ActionNeurons) {
                _neuronLayerDict.Add(neuron.ID, LayerArray.Count);
                LayerArray[LayerArray.Count-1].Add(neuron.ID);
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

            return highestSourceNeuronLayer + 1;
        }
    }
}
