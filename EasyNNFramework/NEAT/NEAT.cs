using System;
using System.Collections.Generic;
using System.Linq;
using EasyNNFramework.FeedForward;

namespace EasyNNFramework {
    [Serializable]
    public class NEAT {

        public Dictionary<string, Neuron> inputNeurons, hiddenNeurons, actionNeurons;
        
        public int counter = 0;
        public int highestLayer = 2;

        public NEAT(Dictionary<string, Neuron> _inputNeurons, Dictionary<string, Neuron> _actionNeurons) {
            inputNeurons = _inputNeurons;
            hiddenNeurons = new Dictionary<string, Neuron>();
            actionNeurons = _actionNeurons;
        }

        public void Mutate(System.Random rnd, float chanceUpdateWeight, float chanceRemoveWeight, float chanceAddNeuron, float chanceRandomFunction) {
            List<Neuron> possibleStartNeurons = new List<Neuron>();
            possibleStartNeurons.AddRange(inputNeurons.Values);
            possibleStartNeurons.AddRange(hiddenNeurons.Values);

            List<Neuron> possibleEndNeurons = new List<Neuron>();
            possibleEndNeurons.AddRange(hiddenNeurons.Values);
            possibleEndNeurons.AddRange(actionNeurons.Values);

            //choose random start neuron
            Neuron rndStart = possibleStartNeurons[rnd.Next(0, possibleStartNeurons.Count)];

            //rnd end neuron
            Neuron rndEnd = possibleEndNeurons[rnd.Next(0, possibleEndNeurons.Count)];

            //differentiate between neurons depending on layer
            //this is necessary when 2 hidden neurons were chosen as random neurons
            Neuron higherLayerNeuron, lowerLayerNeuron;
            if (rndStart.type == NeuronType.Input || rndEnd.type == NeuronType.Action) {
                lowerLayerNeuron = rndStart;
                higherLayerNeuron = rndEnd;
            } else if (rndStart.getLayer(this) > rndEnd.getLayer(this)) {
                higherLayerNeuron = rndStart;
                lowerLayerNeuron = rndEnd;
            } else if (rndStart.getLayer(this) < rndEnd.getLayer(this)) {
                higherLayerNeuron = rndEnd;
                lowerLayerNeuron = rndStart;
            } else {
                lowerLayerNeuron = inputNeurons.Values.ToList()[rnd.Next(0, inputNeurons.Count)];
                higherLayerNeuron = actionNeurons.Values.ToList()[rnd.Next(0, actionNeurons.Count)];
            }

            float weight = WeightHandler.getWeight(lowerLayerNeuron, higherLayerNeuron);
            //if weight to rnd neuron doesnt exist
            if (weight == 0f) {
                weight = UtilityClass.RandomWeight(rnd);
                WeightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, weight);
            }

            //mutation options
            float rndChance = (float)rnd.NextDouble();
            if (rndChance <= chanceUpdateWeight / 100f) {
                if (rnd.NextDouble() <= 0.5f) {
                    //update weight with random value
                    WeightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, UtilityClass.RandomWeight(rnd));
                } else {
                    //update weight with random multiplier
                    WeightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, UtilityClass.Clamp(-4f, 4f, weight * 2f * (float) rnd.NextDouble()));
                }
            } else if (rndChance <= (chanceUpdateWeight + chanceRemoveWeight) / 100f) {
                //remove weight
                WeightHandler.removeWeight(lowerLayerNeuron, higherLayerNeuron);
            } else if (rndChance <= (chanceUpdateWeight + chanceRemoveWeight + chanceAddNeuron) / 100f) {
                //add neuron between start and end
                Neuron newHidden = new Neuron(higherLayerNeuron.getLayer(this) + "hidden" + counter, NeuronType.Hidden,
                    ActivationFunction.GELU);
                hiddenNeurons.Add(newHidden.name, newHidden);
                counter++;
                WeightHandler.removeWeight(lowerLayerNeuron, higherLayerNeuron);
                WeightHandler.addWeight(lowerLayerNeuron, newHidden, weight);
                WeightHandler.addWeight(newHidden, higherLayerNeuron, 1);
            } else if (rndChance <= (chanceUpdateWeight + chanceRemoveWeight + chanceAddNeuron + chanceRandomFunction) / 100f) {
                lowerLayerNeuron.function = getRandomFunction(rnd);
            }

            removeUselessHidden();
            recalculateLayer();
        }

        private void recalculateLayer() {

            int highestHiddenLayer = 1;
            foreach (Neuron hiddenNeuron in hiddenNeurons.Values) {
                int layer = hiddenNeuron.getLayer(this);
                if (layer > highestHiddenLayer) {
                    highestHiddenLayer = layer;
                }
            }

            highestLayer = highestHiddenLayer + 1;
        }

        private void removeUselessHidden() {
            foreach (Neuron hiddenNeuron in hiddenNeurons.Values.ToList()) {
                if (hiddenNeuron.incommingConnections.Count == 0 || hiddenNeuron.outgoingConnections.Count == 0) {
                    foreach (string incoming in hiddenNeuron.incommingConnections.Keys.ToList()) {
                        WeightHandler.removeWeight(getNeuronWithName(incoming), hiddenNeuron);
                    }

                    foreach (string outgoing in hiddenNeuron.outgoingConnections.Keys.ToList()) {
                        WeightHandler.removeWeight(hiddenNeuron, getNeuronWithName(outgoing));
                    }

                    hiddenNeurons.Remove(hiddenNeuron.name);
                }
            }
        }

        public void calculateNetwork() {
            foreach (Neuron actionNeuron in actionNeurons.Values) {
                actionNeuron.calculateValueWithIncomingConnections(this);
            }
            resetNeuronCalculatedValues();
        }

        public Neuron getNeuronWithName(string name) {

            if (inputNeurons.ContainsKey(name)) {
                return inputNeurons[name];
            }
            
            if (hiddenNeurons.ContainsKey(name)) {
                return hiddenNeurons[name];
            }
            
            if (actionNeurons.ContainsKey(name)) {
                return actionNeurons[name];
            }

            return null;
        }

        private void resetNeuronCalculatedValues() {
            foreach (Neuron neuron in hiddenNeurons.Values) {
                neuron.isCalculated = false;
            }

            foreach (Neuron neuron in actionNeurons.Values) {
                neuron.isCalculated = false;
            }
        }

        private ActivationFunction getRandomFunction(Random rnd) {
            return (ActivationFunction)rnd.Next(0, Enum.GetValues(typeof(ActivationFunction)).Length);
        }
    }
}
