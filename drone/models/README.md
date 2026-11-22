# Drone models

From [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), commit `370c43965323c0944b433b202aac5058b5f907a9`.
Not committed: `mujoco_menagerie/` is gitignored. Download only the two drones:

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git drone/models/mujoco_menagerie
git -C drone/models/mujoco_menagerie sparse-checkout set skydio_x2 bitcraze_crazyflie_2
```

| Drone | Mass | Controls | Role |
|---|---|---|---|
| Skydio X2 (`skydio_x2/x2.xml`) | 1.325 kg | 4 rotor thrusts, 0–13 N each | Main intruder |
| Crazyflie 2 (`bitcraze_crazyflie_2/cf2.xml`) | 0.027 kg | total thrust + 3 moments | Unseen drone type for testing |

Fly them yourself: `python drone/view_drones.py` (`--drone x2` or `--drone cf2` for one; `--selftest` checks the flight controller without a window).
