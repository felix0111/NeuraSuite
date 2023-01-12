using System;
using System.Collections.Generic;
using System.Linq;

namespace EasyNNFramework.NEAT {
    [Serializable]
    public static class WeightHandler {

        //updates weight when already added
        public static void addWeight(int sourceID, int targetID, float weight, NEAT network) {

            //if source is action neuron (recurrent weight)
            if (network.layerManager.actionLayer.neurons.ContainsKey(sourceID)) {
                //if weight list from source already exists and weight doesnt exist
                if (network.recurrentConnectionList.TryGetValue(sourceID, out List<Connection> consR) && !consR.Exists(x => x.targetID == targetID)) {
                    network.recurrentConnectionList[sourceID] = consR.Append(new Connection(targetID, weight)).ToList();
                } else {
                    network.recurrentConnectionList.Add(sourceID, new List<Connection>() {new Connection(targetID, weight)});
                }
                return;
            }

            //if normal connection
            //if weight list from source already exists and weight doesnt exist
            if (network.connectionList.TryGetValue(sourceID, out List<Connection> cons) && !cons.Exists(x => x.targetID == targetID)) {
                network.connectionList[sourceID] = cons.Append(new Connection(targetID, weight)).ToList();
            } else {
                network.connectionList.Add(sourceID, new List<Connection>() {new Connection(targetID, weight)});
            }
        }

        public static void updateWeight(int sourceID, int targetID, float weight, NEAT network) {

            //if source is action
            if (network.layerManager.actionLayer.neurons.ContainsKey(sourceID)) {
                //get connections without the connection to update
                List<Connection> consR = network.recurrentConnectionList[sourceID].Where(x => x.targetID != targetID).ToList();
                consR.Add(new Connection(targetID, weight));

                network.recurrentConnectionList[sourceID] = consR;
                return;
            }

            List<Connection> cons = network.connectionList[sourceID].Where(x => x.targetID != targetID).ToList();
            cons.Add(new Connection(targetID, weight));
            network.connectionList[sourceID] = cons;
        }

        public static void removeWeight(int sourceID, int targetID, NEAT network) {

            //if source is action
            if (network.layerManager.actionLayer.neurons.ContainsKey(sourceID)) {
                //get connections without the connection to update
                List<Connection> consR = network.recurrentConnectionList[sourceID].Where(x => x.targetID != targetID).ToList();
                network.recurrentConnectionList[sourceID] = consR;

                //check if weights are empty
                if (network.recurrentConnectionList[sourceID].Count == 0) network.recurrentConnectionList.Remove(sourceID);

                return;
            }

            List<Connection> cons = network.connectionList[sourceID].Where(x => x.targetID != targetID).ToList();
            network.connectionList[sourceID] = cons;

            //check if weights are empty
            if (network.connectionList[sourceID].Count == 0) network.connectionList.Remove(sourceID);
        }

        public static void removeAllConnections(int ID, NEAT network) {

            //check normal connections
            foreach (KeyValuePair<int, List<Connection>> connection in network.connectionList.ToList()) {

                //check if source of connection is ID
                if (connection.Key == ID) {
                    network.connectionList.Remove(connection.Key);
                    continue;
                }

                //remove all connections in list with ID
                network.connectionList[connection.Key] = network.connectionList[connection.Key].Where(x => x.targetID != ID).ToList();

                //remove if empty weight
                if (network.connectionList[connection.Key].Count == 0) network.connectionList.Remove(connection.Key);
            }

            //check recurrent connections
            foreach (KeyValuePair<int, List<Connection>> connection in network.recurrentConnectionList.ToList()) {

                //check if source of connection is ID
                if (connection.Key == ID) {
                    network.recurrentConnectionList.Remove(connection.Key);
                    continue;
                }

                //remove all connections in list with ID
                network.recurrentConnectionList[connection.Key] = network.recurrentConnectionList[connection.Key].Where(x => x.targetID != ID).ToList();

                //remove if empty weight
                if (network.recurrentConnectionList[connection.Key].Count == 0) network.recurrentConnectionList.Remove(connection.Key);
            }
        }
    }

    [Serializable]
    public struct Connection {
        public float weight;
        public int targetID;

        public Connection(int targetID, float weight) {
            this.targetID = targetID;
            this.weight = weight;
        }
    }
}
