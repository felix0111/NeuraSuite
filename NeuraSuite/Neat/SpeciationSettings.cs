using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuraSuite.Neat
{
    public struct SpeciationSettings {

        /// <summary>
        /// How much disjoint genes influence the distance between 2 genomes.
        /// </summary>
        public double DisjointFactor;

        /// <summary>
        /// How much excess genes influence the distance between 2 genomes.
        /// </summary>
        public double ExcessFactor;

        /// <summary>
        /// How much the weight difference of two genomes influence the distance between them.
        /// </summary>
        public double WeightFactor;

        /// <summary>
        /// If the distance of two genomes are smaller than the threshold, they will belong to the same species.
        /// </summary>
        public double GenomeDistanceThreshold;

        public SpeciationSettings(double disjointFactor, double excessFactor, double weightFactor, double genomeDistanceThreshold) {
            DisjointFactor = disjointFactor;
            ExcessFactor = excessFactor;
            WeightFactor = weightFactor;
            GenomeDistanceThreshold = genomeDistanceThreshold;
        }

    }
}
