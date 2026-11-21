# Debt

Quick fixes taken to move fast. One line each: file, what was hacked, what the real fix is.
`/potato-start product` starts from this list.

- sim/episode.py: re-grip at the chosen force after the test lift. A real Franka Hand needs a second `grasp()` call; model that delay and the brief release. Real fix: decide the force from camera features before grasping, or measure the regrasp on hardware.
- sim/episode.py: the wrist F/T site plus Gaussian noise stands in for Franka's joint-torque-based `O_F_ext_hat_K` estimate. Real fix: estimate the external wrench from simulated joint torques the way libfranka does.
- sim/franka.py: torsional friction uses an assumed contact-patch radius of 6 mm (`ContactParams.torsion_radius`), which drives tilt failures. Real fix: calibrate against Franka Hand pad measurements.
- sim/franka.py: the conveyor is static (items spawn at the pick point). Real fix: a moving belt only if belt motion matters for the grasp.
