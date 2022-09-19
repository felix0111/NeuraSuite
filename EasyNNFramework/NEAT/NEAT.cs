using System;
using System.Collections.Generic;
using System.Linq;
using EasyNNFramework.FeedForward;

namespace EasyNNFramework {
    [Serializable]
    public class NEAT {

        public List<Neuron> inputNeurons, hiddenNeurons, actionNeurons;
        
        public int counter = 0;
        public int highestLayer = 2;

        public NEAT(List<Neuron> _inputNeurons, List<Neuron> _actionNeurons) {
            inputNeurons = _inputNeurons;
            hiddenNeurons = new List<Neuron>();
            actionNeurons = _actionNeurons;
        }

        public void Mutate(System.Random rnd, float chanceUpdateWeight, float chanceRemoveWeight, float chanceAddNeuron, float chanceRandomFunction) {
            List<Neuron> possibleStartNeurons = new List<Neuron>();
            possibleStartNeurons.AddRange(inputNeurons);
            possibleStartNeurons.AddRange(hiddenNeurons);

            List<Neuron> possibleEndNeurons = new List<Neuron>();
            possibleEndNeurons.AddRange(hiddenNeurons);
            possibleEndNeurons.AddRange(actionNeurons);

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
                lowerLayerNeuron = inputNeurons[rnd.Next(0, inputNeurons.Count)];
                higherLayerNeuron = actionNeurons[rnd.Next(0, actionNeurons.Count)];
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
                hiddenNeurons.Add(newHidden);
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
            foreach (Neuron hiddenNeuron in hiddenNeurons) {
                int layer = hiddenNeuron.getLayer(this);
                if (layer > highestHiddenLayer) {
                    highestHiddenLayer = layer;
                }
            }

            highestLayer = highestHiddenLayer + 1;
        }

        private void removeUselessHidden() {
            foreach (Neuron hiddenNeuron in hiddenNeurons.ToList()) {
                if (hiddenNeuron.incommingConnections.Count == 0 || hiddenNeuron.outgoingConnections.Count == 0) {
                    foreach (string incoming in hiddenNeuron.incommingConnections.Keys.ToList()) {
                        WeightHandler.removeWeight(getNeuronWithName(incoming), hiddenNeuron);
                    }

                    foreach (string outgoing in hiddenNeuron.outgoingConnections.Keys.ToList()) {
                        WeightHandler.removeWeight(hiddenNeuron, getNeuronWithName(outgoing));
                    }

                    hiddenNeurons.Remove(hiddenNeuron);
                }
            }
        }

        public void calculateNetwork() {
            foreach (Neuron actionNeuron in actionNeurons) {
                actionNeuron.calculateValueWithIncomingConnections(this);
            }
            resetNeuronCalculatedValues();
        }

        private void resetNeuronCalculatedValues() {
            foreach (Neuron neuron in hiddenNeurons) {
                neuron.isCalculated = false;
            }

            foreach (Neuron neuron in actionNeurons) {
                neuron.isCalculated = false;
            }
        }

        public Neuron getNeuronWithName(string name) {
            foreach (Neuron neuron in inputNeurons) {
                if (neuron.name == name) {
                    return neuron;
                }
            }

            foreach (Neuron neuron in hiddenNeurons) {
                if (neuron.name == name) {
                    return neuron;
                }
            }

            foreach (Neuron neuron in actionNeurons) {
                if (neuron.name == name) {
                    return neuron;
                }
            }

            return null;
        }

        private ActivationFunction getRandomFunction(Random rnd) {
            return (ActivationFunction)rnd.Next(0, Enum.GetValues(typeof(ActivationFunction)).Length);
        }
    }
}
