namespace NeuraSuite.Neat
{
    public struct SpeciationSettings {

        /// <summary>
        /// How much disjoint genes influence the distance between 2 genomes.
        /// </summary>
        public double DisjointFactor = 1D;

        /// <summary>
        /// How much excess genes influence the distance between 2 genomes.
        /// </summary>
        public double ExcessFactor = 1D;

        /// <summary>
        /// How much the weight difference of two genomes influence the distance between them.
        /// </summary>
        public double WeightFactor = 0.4D;

        /// <summary>
        /// If the distance of two genomes are smaller than the threshold, they will belong to the same species.
        /// </summary>
        public double GenomeDistanceThreshold = 0.8D;

        public SpeciationSettings(double disjointFactor, double excessFactor, double weightFactor, double genomeDistanceThreshold) {
            DisjointFactor = disjointFactor;
            ExcessFactor = excessFactor;
            WeightFactor = weightFactor;
            GenomeDistanceThreshold = genomeDistanceThreshold;
        }

    }
}
