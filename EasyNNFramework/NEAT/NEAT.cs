using System;
using System.Collections.Generic;
using EasyNNFramework.FeedForward;

namespace EasyNNFramework {
    [Serializable]
    public class NEAT {

        public List<Neuron> inputNeurons, hiddenNeurons, actionNeurons;

        public WeightHandler weightHandler;
        private Random rnd;

        public int highestLayer = 2;

        public NEAT(List<Neuron> _inputNeurons, List<Neuron> _actionNeurons) {
            inputNeurons = _inputNeurons;
            hiddenNeurons = new List<Neuron>();
            actionNeurons = _actionNeurons;
            weightHandler = new WeightHandler(this);
            rnd = new Random();
        }

        public void Mutate() {
            float rndNumber = (float) rnd.NextDouble();

            Neuron rndInput = inputNeurons[rnd.Next(0, inputNeurons.Count)];

            Neuron rndHidden, rndHidden2;
            if (hiddenNeurons.Count != 0) {
                rndHidden = hiddenNeurons[rnd.Next(0, hiddenNeurons.Count)];
                rndHidden2 = hiddenNeurons[rnd.Next(0, hiddenNeurons.Count)];
            } else {
                //if no hidden neurons, use random action neuron
                rndHidden = actionNeurons[rnd.Next(0, actionNeurons.Count)];
                rndHidden2 = actionNeurons[rnd.Next(0, actionNeurons.Count)];
            }
            Neuron rndAction = actionNeurons[rnd.Next(0, actionNeurons.Count)];

            //remove useless hidden
            foreach (Neuron hiddenNeuron in hiddenNeurons) {
                if (hiddenNeuron.incommingConnections.Count == 0 || hiddenNeuron.outgoingConnections.Count == 0) {
                    foreach (string incoming in hiddenNeuron.incommingConnections) {
                        Neuron focused = getNeuronWithName(incoming);
                        weightHandler.removeWeight(focused, hiddenNeuron);
                    }

                    foreach (string outgoing in hiddenNeuron.outgoingConnections) {
                        Neuron focused = getNeuronWithName(outgoing);
                        weightHandler.removeWeight(hiddenNeuron, focused);
                    }
                    hiddenNeurons.Remove(hiddenNeuron);
                }
            }

            //choose random start neuron
            Neuron rndStart;
            if (rnd.NextDouble() <= 0.5f) {
                rndStart = rndInput;
            } else {
                rndStart = rndHidden;
            }

            //rnd end neuron
            Neuron rndEnd;
            if (rnd.NextDouble() <= 0.5f) {
                rndEnd = rndHidden2;
            } else {
                rndEnd = rndAction;
            }

            Neuron higherLayerNeuron, lowerLayerNeuron;
            if (rndStart.getLayer(this) > rndEnd.getLayer(this)) {
                higherLayerNeuron = rndStart;
                lowerLayerNeuron = rndEnd;
            } else if (rndStart.getLayer(this) < rndEnd.getLayer(this)) {
                higherLayerNeuron = rndEnd;
                lowerLayerNeuron = rndStart;
            } else {
                if (rndStart.outgoingConnections.Count != 0) {
                    higherLayerNeuron = getNeuronWithName(rndStart.outgoingConnections[rnd.Next(0, rndStart.outgoingConnections.Count)]);
                    lowerLayerNeuron = rndStart;
                } else if (rndEnd.outgoingConnections.Count != 0) {
                    higherLayerNeuron = getNeuronWithName(rndEnd.outgoingConnections[rnd.Next(0, rndEnd.outgoingConnections.Count)]);
                    lowerLayerNeuron = rndEnd;
                } else {
                    lowerLayerNeuron = inputNeurons[rnd.Next(0, inputNeurons.Count)];
                    higherLayerNeuron = actionNeurons[rnd.Next(0, actionNeurons.Count)];
                }
            }

            //if weight to rnd neuron doesnt exist
            float weight = weightHandler.getWeight(lowerLayerNeuron, higherLayerNeuron);
            if (weight == 0f) {
                weightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, UtilityClass.RandomWeight(rnd));
                Console.WriteLine("Adding weight between " + lowerLayerNeuron.name + " and " + higherLayerNeuron.name);
            } else {
                if (rndNumber <= 0.25f) {
                    //update weight with random value
                    weightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron, UtilityClass.RandomWeight(rnd));
                    Console.WriteLine("Update weight between " + lowerLayerNeuron.name + " and " + higherLayerNeuron.name);
                } else if (rndNumber <= 0.50f) {
                    //update weight with random multiplier
                    weightHandler.addWeight(lowerLayerNeuron, higherLayerNeuron,
                        UtilityClass.InverseLerp(0, 4, Math.Abs(weight) * (float) rnd.NextDouble()) *
                        Math.Sign(weight));
                    Console.WriteLine("Update weight between " + lowerLayerNeuron.name + " and " + higherLayerNeuron.name);
                } else if (rndNumber <= 0.75f) {
                    //remove weight
                    weightHandler.removeWeight(lowerLayerNeuron, higherLayerNeuron);
                    Console.WriteLine("Removing weight between " + lowerLayerNeuron.name + " and " + higherLayerNeuron.name);
                } else {
                    //add neuron between start and end
                    Neuron newHidden = new Neuron(higherLayerNeuron.getLayer(this) + "hidden" + hiddenNeurons.Count, NeuronType.Hidden,
                        ActivationFunction.GELU);
                    hiddenNeurons.Add(newHidden);
                    weightHandler.removeWeight(lowerLayerNeuron, higherLayerNeuron);
                    weightHandler.addWeight(lowerLayerNeuron, newHidden, weight);
                    weightHandler.addWeight(newHidden, higherLayerNeuron, 1);
                    Console.WriteLine("Adding neuron between " + lowerLayerNeuron.name + " and " + higherLayerNeuron.name);
                }
            }
        }

        public void calculateNetwork() {
            foreach (Neuron actionNeuron in actionNeurons) {
                actionNeuron.calculateValueWithIncomingConnections(this);
            }
        }

        public Neuron getNeuronWithName(string name) {

            List<Neuron> neuronList = new List<Neuron>();
            neuronList.AddRange(hiddenNeurons);

            if (!name.Contains("hidden")) {
                neuronList.AddRange(inputNeurons);
                neuronList.AddRange(actionNeurons);
            }

            foreach (Neuron neuron in neuronList) {
                if (neuron.name == name) {
                    return neuron;
                }
            }

            return null;
        }
    }
}
