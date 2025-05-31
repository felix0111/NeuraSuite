namespace NeuraSuite.Neat {

    public struct NeatOptions {

        /// <summary>
        /// Defines the targeted number of genomes after repopulation.
        /// </summary>
        public int TargetPopulationSize;

        /// <summary>
        /// Defines the amount of low-performing genomes that are removed from each species when repopulating.
        /// </summary>
        public double RemoveWorstPercentage;

        /// <summary>
        /// The probability for crossover to happen when creating new offspring. Otherwise offspring is a direct clone.
        /// </summary>
        public double CrossoverChance;

        /// <summary>
        /// The probability for a gene to be enabled when at least one parents gene is disabled.
        /// </summary>
        public double EnableChance;

        /// <summary>
        /// The amount of generations a species fitness can not improve before it counts as stagnant.
        /// When a species counts as stagnant, it might not produce any offspring. 
        /// </summary>
        public int StagnationThreshold;

        /// <summary>
        /// If the distance of two genomes are smaller than the threshold, they will belong to the same species.
        /// </summary>
        public double GenomeDistanceThreshold;

        public NeatOptions(int targetPopulationSize, double removeWorstPercentage, double crossoverChance, double enableChance, int stagnationThreshold, double genomeDistanceThreshold) {
            TargetPopulationSize = targetPopulationSize;
            RemoveWorstPercentage = removeWorstPercentage;
            CrossoverChance = crossoverChance;
            EnableChance = enableChance;
            StagnationThreshold = stagnationThreshold;
            GenomeDistanceThreshold = genomeDistanceThreshold;
        }
    }
}