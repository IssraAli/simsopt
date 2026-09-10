manifold / fixed-point optimization with simsopt + pyoculus

field_topology_optimizables.py is the glue. Chris Smiet wrote it. It wraps pyoculus
fixed points and manifolds as simsopt Optimizables (Greene residue, X-point location,
turnstile area) so you can drop them straight into a LeastSquaresProblem.

setup:
  pip install simsopt
  pip install "git+https://github.com/Akitzu/pyoculus@master"   # need the fork - has the Manifold/turnstile stuff, pypi pyoculus doesn't
  cp field_topology_optimizables.py <your simsopt>/mhd/          # so 'from simsopt.mhd import field_topology_optimizables' works

examples/starter_fixedpoints/ - just finds the fixed points on an LHD-like config, no
  external data. run this first to check the install:
    cd examples/starter_fixedpoints && python simsopt_lhd_like.py

examples/manifold_optimization/ - the actual opt. builds PyOculusFixedPoint targets +
  a LeastSquaresProblem, runs least_squares_serial_solve over the coil/dipole dofs.
  coilset + fixed-point list are in the folder.
    cd examples/manifold_optimization && python NERSC-LHD_opt_w_dipoles_5-31-25.py
  it was set up for NERSC scratch. data paths now default to the script dir; override
  with MANIFOLD_OPT_DATA=... if you move them, and output goes to $SCRATCH if set.
  run_script.sh is the sbatch script I used.
