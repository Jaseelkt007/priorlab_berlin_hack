# Debt

Quick fixes taken to move fast. One line each: file, what was hacked, what the real fix is.
`/potato-start product` starts from this list.

- sim/episode.py: re-grip at the chosen force after the test lift. A real Franka Hand needs a second `grasp()` call; model that delay and the brief release. Real fix: decide the force from camera features before grasping, or measure the regrasp on hardware.
- sim/episode.py: the wrist F/T site plus Gaussian noise stands in for Franka's joint-torque-based `O_F_ext_hat_K` estimate. Real fix: estimate the external wrench from simulated joint torques the way libfranka does.
- sim/franka.py: torsional friction uses an assumed contact-patch radius of 6 mm (`ContactParams.torsion_radius`), which drives tilt failures. Real fix: calibrate against Franka Hand pad measurements.
- sim/franka.py: the conveyor is static (items spawn at the pick point). Real fix: a moving belt only if belt motion matters for the grasp.
- scripts/spike_cudagraph.py: re-compiles `TabPFNV3p5.forward` from its source with the `.item()` class-count sync replaced by a fixed `_fixed_num_classes = 2`, and replaces `model.forward` with a CUDA-graph wrapper found via `gc`. Both depend on TabPFN 10.2.0 internals. Real fix: an upstream option to pass the class count, and a supported hook for graph capture (or the JAX port).
- sim/suction.py: the rigid lip gives no pre-failure warning. The vacuum-sensor reading (seal at touch, dip during the move) will be derived from simulated lip contact forces through an assumed leak curve. Real fix: a compliant bellows cup (spring-jointed lip) or calibration of the leak curve against a real vacuum sensor log.
- sim/suction.py: the delta's arm dynamics are not modelled; the effector tracks the reference exactly. Real fix: closed-chain delta with motor dynamics, only if tracking error matters for grip.
- jax_tabpfn/refit_demo.py: runs torch and JAX in one interpreter by appending the torch env's site-packages to `PYTHONPATH` of `.venv-jax`. Real fix: one env with torch cu132 + `jax[cuda13]` (they coexist on the RTX 4050 as tested), once the main session agrees.
- jax_tabpfn/export.py: supports only 1 estimator, all-numeric columns and TabPFN-3.5's default preprocessing (transform `none`, no categoricals, T=1, no balancing); anything else raises. Real fix: port the remaining TabPFN preprocessing steps (quantile, SVD, categorical encoding) and ensembles.
- jax_tabpfn/runtime.py: the context size is a static shape, so a new context row count compiles once (15–110 s, then persistent cache). Real fix: mask padded train rows in the ICL/decoder attention and pad the ECDF buckets, so contexts can be bucketed like queries.
- jax_tabpfn/model.py: the fast-mode decoder asserts at most 64 classes (one flash-attention call of one-hot values). Real fix: loop over 64-class chunks like `_class_attention`.
