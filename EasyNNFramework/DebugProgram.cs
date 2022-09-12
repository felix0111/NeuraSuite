using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyNNFramework {
    internal static class DebugProgram {
        static void Main(string[] args) {
            FFNN test = new FFNN(1, new List<Neuron>(), 4, new List<Neuron>());
            Console.Read();
        }
    }
}
