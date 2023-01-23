using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using EasyNNFramework.NEAT;

namespace EasyNNFramework {
    public static class UtilityClass {
        public static T DeepClone<T>(this T obj) {
            using (var ms = new MemoryStream()) {
                var formatter = new BinaryFormatter();
                formatter.Serialize(ms, obj);
                ms.Position = 0;

                return (T)formatter.Deserialize(ms);
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

            return (float)rndTrue * rndPos;
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
    }
}
