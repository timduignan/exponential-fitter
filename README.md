# exponential-fitter
Short Jax-MD code for fitting pairwise exponential potentials to ab initio calculations of forces of small clusters.

Loads ab initio calculations of forces on pure water clusters and ion-water clusters as well as the position of the atoms.

The positions were extracted from AIMD simulation of electrolyte solutiosn with CP2K. 

It computes the difference between the forces at the R2SCAN and MP2 level and fits a pair wise exponential potential acting between every pair of atoms to minimise the difference in the forces. 


