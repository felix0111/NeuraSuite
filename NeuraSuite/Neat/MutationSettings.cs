namespace NeuraSuite.Neat
{
    public struct MutationSettings {

        public double AddConnectionChance, SplitConnectionChance, ChangeWeightChance, MaxWeightDelta;

        public MutationSettings(double addConnectionChance, double splitConnectionChance, double changeWeightChance, double maxWeightDelta) {
            AddConnectionChance = addConnectionChance;
            SplitConnectionChance = splitConnectionChance;
            ChangeWeightChance = changeWeightChance;
            MaxWeightDelta = maxWeightDelta;
        }

    }
}
