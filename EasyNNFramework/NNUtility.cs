using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using EasyNNFramework.NEAT;

public static class NNUtility {

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

    public static float InverseLerp(float start, float end, float value) {
        if (start == end) return 0f;
        return (value - start) / (end - start);
    }

    public static float[] InverseLerp(float start, float end, float[] data) {
        var newArr = new float[data.Length];
        for (int i = 0; i < data.Length; i++) {
            newArr[i] = InverseLerp(start, end, data[i]);
            if (float.IsNaN(newArr[i])) newArr[i] = 0f;
        }

        return newArr;
    }

    public static float[] Normalize(float[] data) {
        float[] newArr = new float[data.Length];

        float sum = data.Sum();
        if (sum == 0f) return data.Select(o => 1f / data.Length).ToArray(); //if all data is 0, evenly distribute population

        for (int i = 0; i < data.Length; i++) {
            newArr[i] = data[i] / sum;
        }

        return newArr;
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

    public static ActivationFunction RandomActivationFunction(Random rnd) {
        return (ActivationFunction)rnd.Next(0, Enum.GetValues(typeof(ActivationFunction)).Length);
    }

    public static ActivationFunction RandomActivationFunction(Random rnd, ActivationFunction[] functionPool) {
        return functionPool.Length == 0 ? default : functionPool[rnd.Next(0, functionPool.Length)];
    }

    public static int RandomSign(Random rnd) {
        return rnd.Next(0, 2) * 2 - 1;
    }

    public static float[] Softmax(float[] input) {
        var z = input;

        var z_exp = z.Select(x => Math.Exp(x));
        // [2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09]

        float sum_z_exp = (float)z_exp.Sum();
        // 114.98

        return z_exp.Select(i => (float)i / sum_z_exp).ToArray();
    }

    public static float Latch(float initialValue, float inputValue) {
        if (inputValue >= 1f) {
            return 1f;
        }
        
        if (inputValue < 0f) {
            return 0f;
        }

        return initialValue;
    }

    public static float Gauss(float value) {
        return (float)Math.Pow(Math.E, -(value*value));
    }
}