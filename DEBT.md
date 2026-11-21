# Debt

Quick fixes taken to move fast. One line each: file, what was hacked, what the real fix is.
`/potato-start product` starts from this list.

- sim/episode.py: re-grip at the chosen force after the test lift. A real Franka Hand needs a second `grasp()` call; model that delay and the brief release. Real fix: decide the force from camera features before grasping, or measure the regrasp on hardware.
- sim/episode.py: the wrist F/T site plus Gaussian noise stands in for Franka's joint-torque-based `O_F_ext_hat_K` estimate. Real fix: estimate the external wrench from simulated joint torques the way libfranka does.
- sim/franka.py: torsional friction uses an assumed contact-patch radius of 6 mm (`ContactParams.torsion_radius`), which drives tilt failures. Real fix: calibrate against Franka Hand pad measurements.
- sim/franka.py: the conveyor is static (items spawn at the pick point). Real fix: a moving belt only if belt motion matters for the grasp.
- scripts/spike_cudagraph.py: re-compiles `TabPFNV3p5.forward` from its source with the `.item()` class-count sync replaced by a fixed `_fixed_num_classes = 2`, and replaces `model.forward` with a CUDA-graph wrapper found via `gc`. Both depend on TabPFN 10.2.0 internals. Real fix: an upstream option to pass the class count, and a supported hook for graph capture (or the JAX port).
