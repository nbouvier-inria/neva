## Presentation
This projects contains multiple implementation of different optimization algorithms adapted to spiking neural networks. The neuromorphic evolutionnary algorithm NEVA is implemented both using the LAVA emulator and numpy arrays. You can also install the package to access the algorithm directly.
## Installation
Run "pithon3 -m pip install -e ROOT/neva" to install the package. Be aware that most of the code is inside example files as LAVA has a lot of edge effect and was thus only included in example scripts.
## Contains
- neva.tools contains general tools used in the various optimization algorithms in example files
- example contains most of the code : 
  * example.binary contains binary optimization algorithms
  * example.continuous contains unconstrained continuous optimization algorithms
  * example.permutations contains optimization algorithms effective on permutations
- neva is formatted in the same manner
