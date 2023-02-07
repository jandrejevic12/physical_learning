# Extending LAMMPS for physical learning

## Dowloading LAMMPS
Clone the LAMMPS source code via

```git clone -b release https://github.com/lammps/lammps.git lammps```

## Adding custom routines
Add ```bond_harmonic_learning.h``` and ```bond_harmonic_learning.cpp``` to the ```lammps/src/MOLECULE/``` directory.

Modify the ```lammps/src/Makefile.list``` to include these files.

## Compiling LAMMPS
Make a build folder in the ```lammps``` git directory:
```mkdir my_build```
```cd my_build```

Configure with basic set of packages, including MOLECULE:
```cmake3 -C ../cmake/presets/basic.cmake ../cmake```

Note local command may be ```cmake``` or ```cmake3```.

Compile:
```cmake3 --build .```

An executable named ```lmp``` should be created in the ```lammps/my_build``` directory.

