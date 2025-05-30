﻿using System.Collections.Generic;
using System.Linq;
using NeuraSuite.Neat.Core;

namespace NeuraSuite.Neat.Utility {

    public class InnovationManager {

        //global node counter
        //node ids are zero based
        private int _nodeCounter;

        /// <summary>
        /// Returns a new node id and increments the global counter.
        /// </summary>
        public int NewNodeId => _nodeCounter++;

        //key: (startNodeId, endNodeId) value: innovation
        //innovations are zero-based
        private Dictionary<(int, int), int> _connectionInnovations = new();

        public InnovationManager(Genome initialGenome) {
            foreach (var connection in initialGenome.Connections.Values) {
                _connectionInnovations.Add((connection.StartId, connection.EndId), connection.Innovation);
            }

            _nodeCounter = initialGenome.Nodes.Values.Max(o => o.Id) + 1;
        }

        /// <summary>
        /// Returns the innovation for the specified connection. Returns a new innovation when not found.
        /// </summary>
        public int GetInnovation(int startNode, int endNode) {
            if (_connectionInnovations.TryGetValue((startNode, endNode), out int innov)) return innov;

            int newInnov = _connectionInnovations.Count;
            _connectionInnovations.Add((startNode, endNode), newInnov);
            return newInnov;
        }
    }
}
