# Related work: tabular foundation models for control

## Lee, Takase, Farahat: "An Evaluation of Tabular Foundation Models as Training-Free World Models" (arXiv 2611.11921, Nov 2026)
- **Setup:** off-the-shelf TabPFN **v2** predicts the change in state (next − current) from state and action. Tested on Lorenz, a linear system, a pendulum and **D4RL Hopper/HalfCheetah**, all from offline datasets.
- **Evaluation:** one-step MSE and open-loop rollouts of 20 steps. **No planning, no MPC, no closed-loop control, no adaptation to changed dynamics.**
- **Results:**
  - Beats few-shot LSTM and MLP models given the same amount of data.
  - Models trained on the full dataset are better.
  - With a short context, rollouts are no better than predicting "no change"; with a long context they improve by more than 10×.
- **Latency:** about **3 s per Hopper prediction**. The authors suggest it suits offline use better than "the inner loop of a planner".
- **Open problems they list:** closed-loop evaluation, comparison with dynamics-specific priors, multiple seeds, inference cost.

## ICR-RL: Schiff, Lindenbaum, Efroni (arXiv 2509.11259, ICML 2026)
- TabPFN v2 as a **Q-function trained without gradients**: fitted Q-iteration where the "fitting" happens in context. Classic control only (CartPole, MountainCar, Acrobot).
- Competitive with DQN and better than PPO and TRPO.
- Grows the context online with its best episodes. **Nothing on changed dynamics, and nothing high-dimensional.**

## Not yet read (closest prior art to check)
- OSWM (arXiv 2409.14084): a PFN world model trained on a synthetic prior.
- Benechehab et al.: in-context dynamics learning with LLMs.

## Gap we can claim
Nobody has shown **closed-loop MPC in MuJoCo with a TabPFN-3.5 dynamics model that adapts online, in context, to changed physics**. The binding constraint is latency, which `fit_with_cache`, the Fast checkpoint, a small context and batched candidate evaluation should address. We still need to measure it.
