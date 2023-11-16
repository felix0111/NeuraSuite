using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyNNFramework.NEAT {

    [Serializable]
    public struct Connection : IEquatable<Connection> {

        public float Weight;
        public int TargetID, SourceID, InnovationID;

        public Connection(int innovationID, int sourceID, int targetID, float weight) {
            InnovationID = innovationID;
            SourceID = sourceID;
            TargetID = targetID;
            Weight = weight;
        }

        public override bool Equals(object obj) => obj is Connection n && Equals(n);

        public static bool operator ==(Connection lf, Connection ri) => lf.Equals(ri);

        public static bool operator !=(Connection lf, Connection ri) => !(lf == ri);

        public override int GetHashCode() => InnovationID.GetHashCode();

        public bool Equals(Connection obj) => InnovationID == obj.InnovationID;
    }
}
