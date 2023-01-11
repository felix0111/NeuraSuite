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

        public static Dictionary<int, Neuron> CopyNeuronDictionary(Dictionary<int, Neuron> dict) {
            Dictionary<int, Neuron> copy = new Dictionary<int, Neuron>(dict.Count);
            foreach (Neuron neuron in dict.Values) {
                copy.Add(neuron.ID, new Neuron(neuron.ID, neuron.function));
            }

            return copy;
        }

        public static Dictionary<int, List<Connection>> CopyConnectionDictionary(Dictionary<int, List<Connection>> dict) {
            Dictionary<int, List<Connection>> copy = new Dictionary<int, List<Connection>>(dict.Count);
            foreach (KeyValuePair<int, List<Connection>> pair in dict) {
                copy.Add(pair.Key, new List<Connection>(pair.Value));
            }

            return copy;
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
