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

        List<Neuron> newInputList = new List<Neuron>(tmp.inputNeurons);
        List<Neuron> newActionList = new List<Neuron>(tmp.actionNeurons);
        List<Neuron> newHiddenList = new List<Neuron>(tmp.hiddenNeurons);

        Neuron[] all = newInputList.Concat(newActionList).Concat(newHiddenList).ToArray();

        for (int i = 0; i < all.Length; i++) {

            //if neuron needs to be shifted down
            if (all[i].ID > toRemove.ID) {
                Neuron shiftedNeuron = all[i];
                shiftedNeuron.ID -= 1;

                //shift the connections
                foreach (int sourceID in shiftedNeuron.incommingConnections) {
                    int sourceIndex = Array.FindIndex(all, o => o.ID == sourceID);
                    all[sourceIndex].outgoingConnections.Remove(all[i].ID);
                    all[sourceIndex].outgoingConnections.Add(shiftedNeuron.ID);
                }
                foreach (int targetID in shiftedNeuron.outgoingConnections) {
                    int targetIndex = Array.FindIndex(all, o => o.ID == targetID);
                    all[targetIndex].incommingConnections.Remove(all[i].ID);
                    all[targetIndex].incommingConnections.Add(shiftedNeuron.ID);
                }
                for (int connectionIndex = 0; connectionIndex < tmp.connectionList.Length; connectionIndex++) {
                    Connection con = tmp.connectionList[connectionIndex];
                    if (con.sourceID == all[i].ID) {
                        tmp.connectionList[connectionIndex] = new Connection(shiftedNeuron.ID, con.targetID, con.weight);
                    } else if (con.targetID == all[i].ID) {
                        tmp.connectionList[connectionIndex] = new Connection(con.sourceID, shiftedNeuron.ID, con.weight);
                    }
                }
                for (int rConnectionIndex = 0; rConnectionIndex < tmp.recurrentConnectionList.Length; rConnectionIndex++) {
                    Connection con = tmp.recurrentConnectionList[rConnectionIndex];
                    if (con.sourceID == all[i].ID) {
                        tmp.recurrentConnectionList[rConnectionIndex] = new Connection(shiftedNeuron.ID, con.targetID, con.weight);
                    } else if (con.targetID == all[i].ID) {
                        tmp.recurrentConnectionList[rConnectionIndex] = new Connection(con.sourceID, shiftedNeuron.ID, con.weight);
                    }
                }

                //replace old with shifted neuron
                all[i] = shiftedNeuron;
            }
        }

        //sort neuron collection
        newInputList = all.Where(o => o.type == NeuronType.Input).ToList();
        newActionList = all.Where(o => o.type == NeuronType.Action).ToList();
        newHiddenList = all.Where(o => o.type == NeuronType.Hidden).ToList();

        if (toRemove.type == NeuronType.Input) newInputList.Remove(toRemove);
        if (toRemove.type == NeuronType.Action) newActionList.Remove(toRemove);

        NEAT newNEAT = new NEAT(newInputList.OrderBy(o => o.ID).ToArray(), newActionList.OrderBy(o => o.ID).ToArray()) {
            hiddenNeurons = newHiddenList.ToArray(),
            connectionList = tmp.connectionList,
            recurrentConnectionList = tmp.recurrentConnectionList,
            IDCounter = tmp.IDCounter - 1
        };
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

        List<Neuron> newInputList = new List<Neuron>(oldNEAT.inputNeurons);
        List<Neuron> newActionList = new List<Neuron>(oldNEAT.actionNeurons);
        List<Neuron> newHiddenList = new List<Neuron>(oldNEAT.hiddenNeurons);

        Neuron[] all = newInputList.Concat(newActionList).Concat(newHiddenList).ToArray();

        //also shift hidden neurons to avoid possible conflicts between last action neuron and first hidden neuron (index wise)
        for (int j = all.Length - 1 ; j >= 0; j--) {

            //shift all neurons at ID or above
            if (all[j].ID >= toAdd.ID) {
                Neuron shiftedNeuron = all[j];
                shiftedNeuron.ID += 1;

                //shift the connections
                foreach (int sourceID in shiftedNeuron.incommingConnections) {
                    int sourceIndex = Array.FindIndex(all, o => o.ID == sourceID);
                    all[sourceIndex].outgoingConnections.Remove(all[j].ID);
                    all[sourceIndex].outgoingConnections.Add(shiftedNeuron.ID);
                }
                foreach (int targetID in shiftedNeuron.outgoingConnections) {
                    int targetIndex = Array.FindIndex(all, o => o.ID == targetID);
                    all[targetIndex].incommingConnections.Remove(all[j].ID);
                    all[targetIndex].incommingConnections.Add(shiftedNeuron.ID);
                }
                for (int i = 0; i < tmp.connectionList.Length; i++) {
                    Connection con = tmp.connectionList[i];
                    if (con.sourceID == all[j].ID) {
                        tmp.connectionList[i] = new Connection(shiftedNeuron.ID, con.targetID, con.weight);
                    } else if (con.targetID == all[j].ID) {
                        tmp.connectionList[i] = new Connection(con.sourceID, shiftedNeuron.ID, con.weight);
                    }
                }
                for (int i = 0; i < tmp.recurrentConnectionList.Length; i++) {
                    Connection con = tmp.recurrentConnectionList[i];
                    if (con.sourceID == all[j].ID) {
                        tmp.recurrentConnectionList[i] = new Connection(shiftedNeuron.ID, con.targetID, con.weight);
                    } else if (con.targetID == all[j].ID) {
                        tmp.recurrentConnectionList[i] = new Connection(con.sourceID, shiftedNeuron.ID, con.weight);
                    }
                }

                //replace old with shifted neuron
                all[j] = shiftedNeuron;
            }
        }

        //sort neuron collection
        newInputList = all.Where(o => o.type == NeuronType.Input).ToList();
        newActionList = all.Where(o => o.type == NeuronType.Action).ToList();
        newHiddenList = all.Where(o => o.type == NeuronType.Hidden).ToList();

        if(toAdd.type == NeuronType.Input) newInputList.Add(toAdd);
        if(toAdd.type == NeuronType.Action) newActionList.Add(toAdd);

        NEAT newNEAT = new NEAT(newInputList.OrderBy(o => o.ID).ToArray(), newActionList.OrderBy(o => o.ID).ToArray()) {
            hiddenNeurons = newHiddenList.ToArray(),
            connectionList = tmp.connectionList,
            recurrentConnectionList = tmp.recurrentConnectionList,
            IDCounter = tmp.IDCounter+1
        };
        return newNEAT;
    }
}