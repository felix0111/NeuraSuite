using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

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

        public static float InverseLerp(float start, float end, float value) {
            return (value - start) / (end - start);
        }

        public static float RandomWeight(Random rnd) {
            float rndPos = (float) rnd.NextDouble() * 4f;
            int rndTrue = rnd.Next(0, 2) * 2 - 1;

            return (float)rndTrue * rndPos;
        }
    }
}
