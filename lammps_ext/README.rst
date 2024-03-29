Extending LAMMPS for physical learning
======================================

1. Clone a local copy of the LAMMPS repository:

.. code-block:: bash
   
   $ git clone -b release https://github.com/lammps/lammps.git lammps

2. Add the provided :code:`bond_harmonic_learning.h/.cpp` and :code:`bond_harmonic_learning_symmetric.h/.cpp` to the :code:`lammps/src/MOLECULE/` directory of the LAMMPS code. Modify :code:`lammps/src/Makefile.list` to include these files in the :code:`SRC` (for .cpp files) and :code:`INC` (for .h files) fields, as shown in the example :code:`Makefile.list` provided.

3. Make a build folder in the `lammps` directory:

.. code-block:: bash
   
   $ mkdir my_build
   $ cd my_build

4. Configure with the basic set of packages, including MOLECULE. Either :code:`cmake` or :code:`cmake3` is necessary:

.. code-block:: bash

   $ cmake3 -C ../cmake/presets/basic.cmake ../cmake

5. Compile the LAMMPS executable. An executable named :code:`lmp` will be created in the :code:`lammps/my_build` directory, which now has the physical learning functionality.

.. code-block:: bash

   $ cmake3 --build .


