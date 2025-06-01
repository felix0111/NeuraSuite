using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuraSuite.Neat
{
    public struct SpeciationSettings {

        /// <summary>
        /// If the distance of two genomes are smaller than the threshold, they will belong to the same species.
        /// </summary>
        public double GenomeDistanceThreshold;

        public SpeciationSettings(double genomeDistanceThreshold) {
            GenomeDistanceThreshold = genomeDistanceThreshold;
        }

    }
}
