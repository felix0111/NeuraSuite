using System;
using System.Collections.Generic;
using System.Linq;
using EasyNNFramework.FeedForward;

namespace EasyNNFramework {
    [Serializable]
    public class NEAT {

        public List<Neuron> inputNeurons, hiddenNeurons, actionNeurons;

        public WeightHandler weightHandler;
        private Random rnd;
        private int counter = 0;

        public int highestLayer = 2;

        public NEAT(List<Neuron> _inputNeurons, List<Neuron> _actionNeurons) {
            inputNeurons = _inputNeurons;
            hiddenNeurons = new List<Neuron>();
            actionNeurons = _actionNeurons;
            weightHandler = new WeightHandler();
            rnd = new Random();
        }

        public void Mutate() {
            float rndNumber = (float) rnd.NextDouble();

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
            } else if(rndStart.getLayer(this) > rndEnd.getLayer(this)) {
                higherLayerNeuron = rndStart;
                lowerLayerNeuron = rndEnd;
            } else if (rndStart.getLayer(this) < rndEnd.getLayer(this)) {
                higherLayerNeuron = rndEnd;
                lowerLayerNeuron = rndStart;
            } else {
                //should normally not be called because hidden neurons always have one connection
                /*if (rndStart.outgoingConnections.Count != 0) {
                    higherLayerNeuron = getNeuronWithName(rndStart.outgoingConnections[rnd.Next(0, rndStart.outgoingConnections.Count)]);
                    lowerLayerNeuron = rndStart;
                } else if (rndEnd.outgoingConnections.Count != 0) {
                    higherLayerNeuron = getNeuronWithName(rndEnd.outgoingConnections[rnd.Next(0, rndEnd.outgoingConnections.Count)]);
                    lowerLayerNeuron = rndEnd;
                } else {*/
                lowerLayerNeuron = inputNeurons[rnd.Next(0, inputNeurons.Count)];
                higherLayerNeuron = actionNeurons[rnd.Next(0, actionNeurons.Count)];
                //}
            }

            float weight = weightHandler.getWeight(lowerLayerNeuron, higherLayerNeuron);
            //if weight to rnd neuron doesnt exist
            if (weight == 0f) {
                weightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, UtilityClass.RandomWeight(rnd));
            } else {
                if (rndNumber <= 0.25f) {
                    //update weight with random value
                    weightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, UtilityClass.RandomWeight(rnd));
                } else if (rndNumber <= 0.50f) {
                    //update weight with random multiplier
                    weightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron,
                        UtilityClass.InverseLerp(0, 4, Math.Abs(weight) * (float) rnd.NextDouble()) *
                        Math.Sign(weight));
                } else if (rndNumber <= 0.75f) {
                    //remove weight
                    weightHandler.removeWeight(lowerLayerNeuron, higherLayerNeuron);
                } else {
                    //add neuron between start and end
                    Neuron newHidden = new Neuron(higherLayerNeuron.getLayer(this) + "hidden" + counter, NeuronType.Hidden,
                        ActivationFunction.GELU);
                    hiddenNeurons.Add(newHidden);
                    counter++;
                    weightHandler.removeWeight(lowerLayerNeuron, higherLayerNeuron);
                    weightHandler.addWeight(lowerLayerNeuron, newHidden, weight);
                    weightHandler.addWeight(newHidden, higherLayerNeuron, 1);
                }
            }

            //remove useless hidden
            foreach (Neuron hiddenNeuron in hiddenNeurons.ToList()) {
                if (hiddenNeuron.incommingConnections.Count == 0 || hiddenNeuron.outgoingConnections.Count == 0) {
                    foreach (string incoming in hiddenNeuron.incommingConnections.Keys.ToList()) {
                        Neuron focused = getNeuronWithName(incoming);
                        weightHandler.removeWeight(focused, hiddenNeuron);
                    }

                    foreach (string outgoing in hiddenNeuron.outgoingConnections.Keys.ToList()) {
                        Neuron focused = getNeuronWithName(outgoing);
                        weightHandler.removeWeight(hiddenNeuron, focused);
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

        public void resetNeuronCalculatedValues() {
            List<Neuron> neuronList = new List<Neuron>();
            
            neuronList.AddRange(actionNeurons);
            neuronList.AddRange(hiddenNeurons);

            foreach (Neuron neuron in neuronList) {
                neuron.isCalculated = false;
            }
        }

        public Neuron getNeuronWithName(string name) {

            List<Neuron> neuronList = new List<Neuron>();
            
            neuronList.AddRange(inputNeurons);
            neuronList.AddRange(actionNeurons);
            neuronList.AddRange(hiddenNeurons);

            foreach (Neuron neuron in neuronList) {
                if (neuron.name == name) {
                    return neuron;
                }
            }

            return null;
        }
    }
}
