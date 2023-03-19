using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyNNFramework.NEAT {

    [Serializable]
    public struct LayerStructure {
        public List<int[]> layerArray;
        private Dictionary<int, int> neuronLayerDict;

        public LayerStructure(in Network network) {
            layerArray = new List<int[]>();
            neuronLayerDict = new Dictionary<int, int>(network.Neurons.Count); //key = neuron id, value = layer

            //calc layer for input and hidden neurons
            foreach (var input in network.InputNeurons) {
                neuronLayerDict.Add(input.ID, 1);
            }
            foreach (Neuron n in network.HiddenNeurons) {
                GetLayer(n.ID, network);
            }

            int maxLayer = neuronLayerDict.Values.Max() + 1;
            //action neurons must be manually set because some may be unconnected
            foreach (var neuron in network.ActionNeurons) {
                neuronLayerDict.Add(neuron.ID, maxLayer);
            }

            //add layers
            for (int i = 1; i <= neuronLayerDict.Values.Max(); i++) {
                var allNeuronsInLayerI = neuronLayerDict.Where(o => o.Value == i).Select(o => o.Key).ToArray();
                layerArray.Add(allNeuronsInLayerI);
            }
        }

        public int GetLayer(int neuronID, in Network network) {
            if (neuronLayerDict.TryGetValue(neuronID, out int v)) return v;

            int highestLayer = 1;
            for (int i = 0; i < network.Neurons[neuronID].IncommingConnections.Count; i++) {
                Connection con = network.Connections[network.Neurons[neuronID].IncommingConnections[i]];
                int l = GetLayer(con.SourceID, network);
                if (l > highestLayer) highestLayer = l;
            }

            neuronLayerDict.Add(neuronID, highestLayer + 1);
            return highestLayer + 1;
        }
    }
}
