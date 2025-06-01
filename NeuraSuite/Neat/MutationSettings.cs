namespace NeuraSuite.Neat
{
    public struct MutationSettings {

        /// <summary>
        /// The chance that two nodes connect.
        /// </summary>
        public double AddConnectionChance = 0.05D;

        /// <summary>
        /// The chance that a new node gets created by splitting a connection.
        /// </summary>
        public double SplitConnectionChance = 0.03D;

        /// <summary>
        /// The chance a weight gets changed.
        /// </summary>
        public double ChangeWeightChance = 0.8D;

        /// <summary>
        /// The maximum amount a weight can change with mutation.
        /// </summary>
        public double MaxWeightDelta = 0.1D;

        public MutationSettings(double addConnectionChance, double splitConnectionChance, double changeWeightChance, double maxWeightDelta) {
            AddConnectionChance = addConnectionChance;
            SplitConnectionChance = splitConnectionChance;
            ChangeWeightChance = changeWeightChance;
            MaxWeightDelta = maxWeightDelta;
        }

    }
}
