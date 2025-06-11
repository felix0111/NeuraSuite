namespace NeuraSuite.Neat {

    public struct NeatSettings {

        /// <summary>
        /// Defines the targeted number of genomes after repopulation.
        /// </summary>
        public int TargetPopulationSize = 150;

        /// <summary>
        /// Defines the amount of low-performing genomes that are removed from each species when repopulating.
        /// </summary>
        public double RemoveWorstPercentage = 0D;

        /// <summary>
        /// The probability for crossover to happen when creating new offspring. Otherwise offspring is a direct clone.
        /// </summary>
        public double CrossoverChance = 0.75D;

        /// <summary>
        /// The probability for a gene to be enabled when at least one parents gene is disabled.
        /// </summary>
        public double EnableChance = 0.25D;

        /// <summary>
        /// The amount of generations a species fitness can not improve before it counts as stagnant.
        /// When a species counts as stagnant, it might not produce any offspring. 
        /// </summary>
        public int SpeciesStagnationThreshold = 100;

        public NeatSettings(int targetPopulationSize, double removeWorstPercentage, double crossoverChance, double enableChance, int speciesStagnationThreshold) {
            TargetPopulationSize = targetPopulationSize;
            RemoveWorstPercentage = removeWorstPercentage;
            CrossoverChance = crossoverChance;
            EnableChance = enableChance;
            SpeciesStagnationThreshold = speciesStagnationThreshold;
        }
    }
}