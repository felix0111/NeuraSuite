using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Threading;
using EasyNNFramework.NEAT;

public static class UtilityClass {
    public static T DeepClone<T>(this T obj) {
        using (var ms = new MemoryStream()) {
            var formatter = new BinaryFormatter();
            formatter.Serialize(ms, obj);
            ms.Position = 0;

            return (T) formatter.Deserialize(ms);
        }
    }

    public static T BinarySearch<T, TKey>(this IList<T> list, Func<T, TKey> keySelector, TKey key)
        where TKey : IComparable<TKey> {
        if (list.Count == 0)
            throw new InvalidOperationException("Item not found");

        int min = 0;
        int max = list.Count;
        while (min < max) {
            int mid = min + ((max - min) / 2);
            T midItem = list[mid];
            TKey midKey = keySelector(midItem);
            int comp = midKey.CompareTo(key);
            if (comp < 0) {
                min = mid + 1;
            } else if (comp > 0) {
                max = mid - 1;
            } else {
                return midItem;
            }
        }

        if (min == max &&
            min < list.Count &&
            keySelector(list[min]).CompareTo(key) == 0) {
            return list[min];
        }

        throw new InvalidOperationException("Item not found");
    }

    public static Neuron[] CopyNeuronArray(this Neuron[] obj) {
        Neuron[] cpy = new Neuron[obj.Length];
        Neuron buffer;
        for (int i = 0; i < obj.Length; i++) {
            buffer = obj[i];
            buffer.incommingConnections = new List<int>(buffer.incommingConnections);
            buffer.outgoingConnections = new List<int>(buffer.outgoingConnections);
            cpy[i] = buffer;
        }

        return cpy;
    }

    public static float InverseLerp(float start, float end, float value) {
        return (value - start) / (end - start);
    }

    public static float RandomWeight(Random rnd) {
        float rndPos = (float) rnd.NextDouble() * 4f;
        int rndTrue = rnd.Next(0, 2) * 2 - 1;

        return (float) rndTrue * rndPos;
    }

    public static float Clamp(float min, float max, float value) {
        if (value > min && value < max) {
            return value;
        }

        if (value < min) {
            return min;
        }

        return max;
    }

    public static NEAT RemoveStaticNeurons(in NEAT oldNEAT, Neuron[] removedNeurons) {
        NEAT tmp = new NEAT(oldNEAT);

        foreach (Neuron n in removedNeurons) {
            if (n.type == NeuronType.Hidden) throw new Exception("Hidden neuron with ID " + n.ID + " is not a static neuron!");

            tmp = RemoveStaticNeuron(tmp, n);
        }

        return tmp;
    }

    public static NEAT RemoveStaticNeuron(in NEAT oldNEAT, Neuron toRemove) {
        if (toRemove.type == NeuronType.Hidden) throw new Exception("Hidden neuron with ID " + toRemove.ID + " is not a static neuron!");

        NEAT tmp = new NEAT(oldNEAT);

        //remove all connections depending on neuron and free up it's ID
        tmp.RemoveDependingConnections(toRemove.ID);
        tmp.recalculateStructure();

        List<Neuron> newInputList = new List<Neuron>();
        List<Neuron> newActionList = new List<Neuron>();

        bool shift = false;
        foreach (Neuron oldInput in tmp.ROinputNeurons.Concat(tmp.ROactionNeurons)) {
            if (oldInput.ID == toRemove.ID) {
                //if reached the neuron to remove, start shifting
                shift = true;
            } else if(shift) {     //if shifting is active
                Neuron shiftedNeuron = oldInput;
                shiftedNeuron.ID -= 1;

                //shift the connections
                foreach (int source in shiftedNeuron.incommingConnections) {
                    tmp.GetNeuronRef(source).outgoingConnections.Remove(oldInput.ID);
                    tmp.GetNeuronRef(source).outgoingConnections.Add(shiftedNeuron.ID);
                }
                foreach (int target in shiftedNeuron.outgoingConnections) {
                    tmp.GetNeuronRef(target).incommingConnections.Remove(oldInput.ID);
                    tmp.GetNeuronRef(target).incommingConnections.Add(shiftedNeuron.ID);
                }
                for (int i = 0; i < tmp.connectionList.Length; i++) {
                    Connection con = tmp.connectionList[i];
                    if (con.sourceID == oldInput.ID) {
                        tmp.connectionList[i] = new Connection(shiftedNeuron.ID, con.targetID, con.weight);
                    } else if (con.targetID == oldInput.ID) {
                        tmp.connectionList[i] = new Connection(con.sourceID, shiftedNeuron.ID, con.weight);
                    }
                }
                for (int i = 0; i < tmp.recurrentConnectionList.Length; i++) {
                    Connection con = tmp.recurrentConnectionList[i];
                    if (con.sourceID == oldInput.ID) {
                        tmp.recurrentConnectionList[i] = new Connection(shiftedNeuron.ID, con.targetID, con.weight);
                    } else if (con.targetID == oldInput.ID) {
                        tmp.recurrentConnectionList[i] = new Connection(con.sourceID, shiftedNeuron.ID, con.weight);
                    }
                }

                if (shiftedNeuron.type == NeuronType.Input) {
                    newInputList.Add(shiftedNeuron);
                } else {
                    newActionList.Add(shiftedNeuron);
                }
                
            } else {    //if neuron must not be shifted, transfer neuron
                if (oldInput.type == NeuronType.Input) {
                    newInputList.Add(oldInput);
                } else {
                    newActionList.Add(oldInput);
                }
            }
        }

        NEAT newNEAT = new NEAT(newInputList.ToArray(), newActionList.ToArray());
        newNEAT.hiddenNeurons = tmp.hiddenNeurons;
        newNEAT.connectionList = tmp.connectionList;
        newNEAT.recurrentConnectionList = tmp.recurrentConnectionList;
        newNEAT.IDCounter = tmp.IDCounter;
        return newNEAT;
    }

    public static NEAT AddStaticNeurons(in NEAT oldNEAT, Neuron[] addedNeurons) {
        NEAT tmp = new NEAT(oldNEAT);

        foreach (Neuron n in addedNeurons) {
            if (n.type == NeuronType.Hidden) throw new Exception("Hidden neuron with ID " + n.ID + " is not a static neuron!");

            tmp = AddStaticNeuron(tmp, n);
        }

        return tmp;
    }

    public static NEAT AddStaticNeuron(in NEAT oldNEAT, Neuron toAdd) {
        if (toAdd.type == NeuronType.Hidden) throw new Exception("Hidden neuron with ID " + toAdd.ID + " is not a static neuron!");

        NEAT tmp = new NEAT(oldNEAT);

        List<Neuron> newInputList = new List<Neuron>();
        List<Neuron> newActionList = new List<Neuron>();
        
        //also shift hidden neurons to avoid possible conflicts between last action neuron and first hidden neuron (index wise)
        foreach (Neuron oldInput in tmp.ROinputNeurons.Concat(tmp.ROactionNeurons).Concat(tmp.hiddenNeurons).Reverse()) {
            //shift all neurons at ID or above
            if (oldInput.ID >= toAdd.ID) {
                Neuron shiftedNeuron = oldInput;
                shiftedNeuron.ID += 1;

                //shift the connections
                foreach (int source in shiftedNeuron.incommingConnections) {
                    tmp.GetNeuronRef(source).outgoingConnections.Remove(oldInput.ID);
                    tmp.GetNeuronRef(source).outgoingConnections.Add(shiftedNeuron.ID);
                }
                foreach (int target in shiftedNeuron.outgoingConnections) {
                    tmp.GetNeuronRef(target).incommingConnections.Remove(oldInput.ID);
                    tmp.GetNeuronRef(target).incommingConnections.Add(shiftedNeuron.ID);
                }
                for (int i = 0; i < tmp.connectionList.Length; i++) {
                    Connection con = tmp.connectionList[i];
                    if (con.sourceID == oldInput.ID) {
                        tmp.connectionList[i] = new Connection(shiftedNeuron.ID, con.targetID, con.weight);
                    } else if (con.targetID == oldInput.ID) {
                        tmp.connectionList[i] = new Connection(con.sourceID, shiftedNeuron.ID, con.weight);
                    }
                }
                for (int i = 0; i < tmp.recurrentConnectionList.Length; i++) {
                    Connection con = tmp.recurrentConnectionList[i];
                    if (con.sourceID == oldInput.ID) {
                        tmp.recurrentConnectionList[i] = new Connection(shiftedNeuron.ID, con.targetID, con.weight);
                    } else if (con.targetID == oldInput.ID) {
                        tmp.recurrentConnectionList[i] = new Connection(con.sourceID, shiftedNeuron.ID, con.weight);
                    }
                }

                //add shifted neuron to list
                if (shiftedNeuron.type == NeuronType.Input) {
                    newInputList.Add(shiftedNeuron);
                } else {
                    newActionList.Add(shiftedNeuron);
                }

                //if new neuron ID is at this position, also add
                if (oldInput.ID == toAdd.ID) {
                    if (shiftedNeuron.type == NeuronType.Input) {
                        newInputList.Add(toAdd);
                    } else {
                        newActionList.Add(toAdd);
                    }
                }

            } else {    //if neuron must not be shifted, transfer neuron
                if (oldInput.type == NeuronType.Input) {
                    newInputList.Add(oldInput);
                } else {
                    newActionList.Add(oldInput);
                }
            }
        }

        newInputList.Reverse();
        newActionList.Reverse();

        NEAT newNEAT = new NEAT(newInputList.ToArray(), newActionList.ToArray());
        newNEAT.hiddenNeurons = tmp.hiddenNeurons;
        newNEAT.connectionList = tmp.connectionList;
        newNEAT.recurrentConnectionList = tmp.recurrentConnectionList;
        newNEAT.IDCounter = tmp.IDCounter+1;
        return newNEAT;
    }
}