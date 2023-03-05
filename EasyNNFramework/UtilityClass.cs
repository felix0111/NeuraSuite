using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Threading;
using EasyNNFramework.NEAT;

public static class UtilityClass {

    public static NEAT Crossover(in NEAT brain1, in NEAT brain2) {
        throw new NotImplementedException();
    }

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
            cpy[i] = obj[i].Clone();
        }

        return cpy;
    }

    public static Connection[] CopyConnectionArray(this Connection[] obj) {
        Connection[] cpy = new Connection[obj.Length];
        for (int i = 0; i < obj.Length; i++) {
            cpy[i] = obj[i];
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
    /*
    public static NEAT RemoveStaticNeurons(in NEAT oldNEAT, Neuron[] removedNeurons) {
        NEAT tmp = new NEAT(oldNEAT);

        foreach (Neuron n in removedNeurons) {
            if (n.Type == NeuronType.Hidden) throw new Exception("Hidden neuron with ID " + n.ID + " is not a static neuron!");

            tmp = RemoveStaticNeuron(tmp, n);
        }

        return tmp;
    }

    public static NEAT RemoveStaticNeuron(in NEAT oldNEAT, Neuron toRemove) {
        if (toRemove.Type == NeuronType.Hidden) throw new Exception("Hidden neuron with ID " + toRemove.ID + " is not a static neuron!");

        NEAT tmp = new NEAT(oldNEAT);

        //remove all connections depending on neuron and free up it's ID
        tmp.RemoveDependingConnections(toRemove.ID, true);
        tmp.RecalculateStructure();

        List<Neuron> newInputList = new List<Neuron>(tmp.InputNeurons);
        List<Neuron> newActionList = new List<Neuron>(tmp.ActionNeurons);
        List<Neuron> newHiddenList = new List<Neuron>(tmp.HiddenNeurons);

        newInputList.RemoveAll(o => o.ID == toRemove.ID);
        newActionList.RemoveAll(o => o.ID == toRemove.ID);

        Neuron[] all = newInputList.Concat(newActionList).Concat(newHiddenList).ToArray();

        for (int i = 0; i < all.Length; i++) {

            //if neuron needs to be shifted down
            if (all[i].ID > toRemove.ID) {
                Neuron shiftedNeuron = all[i];
                shiftedNeuron.SetID(shiftedNeuron.ID-1);

                //shift the connections
                foreach (int sourceID in shiftedNeuron.IncommingConnections) {
                    int sourceIndex = Array.FindIndex(all, o => o.ID == sourceID);
                    all[sourceIndex].OutgoingConnections.Remove(all[i].ID);
                    all[sourceIndex].OutgoingConnections.Add(shiftedNeuron.ID);
                }
                foreach (int targetID in shiftedNeuron.OutgoingConnections) {
                    int targetIndex = Array.FindIndex(all, o => o.ID == targetID);
                    all[targetIndex].IncommingConnections.Remove(all[i].ID);
                    all[targetIndex].IncommingConnections.Add(shiftedNeuron.ID);
                }
                for (int connectionIndex = 0; connectionIndex < tmp.Connections.Count; connectionIndex++) {
                    Connection con = tmp.Connections.ElementAt(connectionIndex).Value;
                    if (con.SourceID == all[i].ID) {
                        tmp.Connections.Remove(con.ID);
                        tmp.Connections.Add(con.ID, new Connection(con.ID, shiftedNeuron.ID, con.TargetID, con.Weight));
                    } else if (con.TargetID == all[i].ID) {
                        tmp.connectionList[connectionIndex] = new Connection(con.SourceID, shiftedNeuron.ID, con.Weight);
                    }
                }
                for (int rConnectionIndex = 0; rConnectionIndex < tmp.recurrentConnectionList.Length; rConnectionIndex++) {
                    Connection con = tmp.recurrentConnectionList[rConnectionIndex];
                    if (con.SourceID == all[i].ID) {
                        tmp.recurrentConnectionList[rConnectionIndex] = new Connection(shiftedNeuron.ID, con.TargetID, con.Weight);
                    } else if (con.TargetID == all[i].ID) {
                        tmp.recurrentConnectionList[rConnectionIndex] = new Connection(con.SourceID, shiftedNeuron.ID, con.Weight);
                    }
                }

                //replace old with shifted neuron
                all[i] = shiftedNeuron;
            } else if (all[i].ID == toRemove.ID) {

            }
        }

        //sort neuron collection
        newInputList = all.Where(o => o.Type == NeuronType.Input).ToList();
        newActionList = all.Where(o => o.Type == NeuronType.Action).ToList();
        newHiddenList = all.Where(o => o.Type == NeuronType.Hidden).ToList();

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
            if (n.Type == NeuronType.Hidden) throw new Exception("Hidden neuron with ID " + n.ID + " is not a static neuron!");

            tmp = AddStaticNeuron(tmp, n);
        }

        return tmp;
    }

    public static NEAT AddStaticNeuron(in NEAT oldNEAT, Neuron toAdd) {
        if (toAdd.Type == NeuronType.Hidden) throw new Exception("Hidden neuron with ID " + toAdd.ID + " is not a static neuron!");

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
                foreach (int sourceID in shiftedNeuron.IncommingConnections) {
                    int sourceIndex = Array.FindIndex(all, o => o.ID == sourceID);
                    all[sourceIndex].OutgoingConnections.Remove(all[j].ID);
                    all[sourceIndex].OutgoingConnections.Add(shiftedNeuron.ID);
                }
                foreach (int targetID in shiftedNeuron.OutgoingConnections) {
                    int targetIndex = Array.FindIndex(all, o => o.ID == targetID);
                    all[targetIndex].IncommingConnections.Remove(all[j].ID);
                    all[targetIndex].IncommingConnections.Add(shiftedNeuron.ID);
                }
                for (int i = 0; i < tmp.connectionList.Length; i++) {
                    Connection con = tmp.connectionList[i];
                    if (con.SourceID == all[j].ID) {
                        tmp.connectionList[i] = new Connection(shiftedNeuron.ID, con.TargetID, con.Weight);
                    } else if (con.TargetID == all[j].ID) {
                        tmp.connectionList[i] = new Connection(con.SourceID, shiftedNeuron.ID, con.Weight);
                    }
                }
                for (int i = 0; i < tmp.recurrentConnectionList.Length; i++) {
                    Connection con = tmp.recurrentConnectionList[i];
                    if (con.SourceID == all[j].ID) {
                        tmp.recurrentConnectionList[i] = new Connection(shiftedNeuron.ID, con.TargetID, con.Weight);
                    } else if (con.TargetID == all[j].ID) {
                        tmp.recurrentConnectionList[i] = new Connection(con.SourceID, shiftedNeuron.ID, con.Weight);
                    }
                }

                //replace old with shifted neuron
                all[j] = shiftedNeuron;
            }
        }

        //sort neuron collection
        newInputList = all.Where(o => o.Type == NeuronType.Input).ToList();
        newActionList = all.Where(o => o.Type == NeuronType.Action).ToList();
        newHiddenList = all.Where(o => o.Type == NeuronType.Hidden).ToList();

        if(toAdd.Type == NeuronType.Input) newInputList.Add(toAdd);
        if(toAdd.Type == NeuronType.Action) newActionList.Add(toAdd);

        NEAT newNEAT = new NEAT(newInputList.OrderBy(o => o.ID).ToArray(), newActionList.OrderBy(o => o.ID).ToArray()) {
            hiddenNeurons = newHiddenList.ToArray(),
            connectionList = tmp.connectionList,
            recurrentConnectionList = tmp.recurrentConnectionList,
            IDCounter = tmp.IDCounter+1
        };
        return newNEAT;
    }*/
}