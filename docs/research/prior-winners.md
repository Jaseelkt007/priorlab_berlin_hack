# Previous Prior Labs hackathon (TabPFN-3.5, about 26 Sep to 4 Oct 2026): what won and why

Source: platform.priorlabs.ai/hackathon-3.5. We read the repos but ran none of the code.

## 1st place: sEMG gesture detection (simon-bachhuber/tabpfn-sEMG-human-gesture-detection)
- **Task:** classify 8 hand gestures from 24 EMG channels in an armband.
- **Setup:** one row is a window of about 0.5 s, described by 144 standard features. `tabpfn-client` 3.5 Plus, 8 estimators, no fine-tuning.
- **Data:** 215k context rows from 40 people; tested on **4 people the model never saw**, with no calibration per user.
- **Result:** 84.7% accuracy, against 82.1% for LDA and 77.8% for SVM. Those baselines are the **dataset authors' own published pipelines, rebuilt exactly**. Per-person results are reported, including the one person where TabPFN loses.
- **Rigour:** pinned upstream commits, a dataset snapshot with SHA-256 hashes, a provenance file, tests, a verification log from a fresh clone, and a replay UI.
- **Why it won:** a real problem, generalization to unseen people, fair published baselines, and close to excessive reproducibility. **Credibility beat spectacle.**

## 2nd place: duckpfn (nikikilbertus/duckpfn). Closest to our plan
- **Setup:** MuJoCo with Pollen's Microduck (14 joints, 61 observations, 50 Hz).
  - TabPFN 3.5 Fast regressor with `fit_with_cache` performs **behaviour cloning of Pollen's pretrained RL policy**.
  - Each row is the observation plus a joint ID, and the target is that joint's action. 4,096 transitions become 57k context rows.
  - Rollouts are closed-loop and the context is frozen during them.
- **Extra experiments:**
  - **Teaching:** swapping in demonstrations made with weaker motors took walks from 0/3 to 3/3 successful (n=3).
  - **Risk-aware posture choice:** choosing poses by the 5th-percentile (q05) prediction gave 56 falls vs 81 for the mean (1,024 pushes).
  - **Fall warning:** a classifier warns 0.14 s before a fall.
  - **Walking from pixels:** about 6 s of walking from PCA features, about 2 s from raw pixels.
  - **Text-to-movement classification:** 51/54 correct.
- **Presentation (its strongest part):**
  - A header GIF, a YouTube film, and a Colab notebook.
  - Interactive widgets that replay recorded results, so "Run All" works with no weights or key.
  - A README written as a table of TabPFN features.
- **Weaknesses:**
  - **No baselines at all.**
  - Tiny and cherry-picked evaluation.
  - Uncertainty admitted to be uncalibrated.
  - **Frozen context, so no online learning.**
  - It only imitates an expert and never measures how close it gets.
  - Data-collection scripts are missing, so the 4,096 transitions cannot be regenerated.
  - Only one walking command was tested.

## 3rd place: Tavolio (Christopher-Wang/tavolio-extension)
- A Google Sheets and Excel sidebar that fills empty cells.
- Runs TabPFN **in the browser** (INT4 ONNX on WebGPU) or through the API, and includes SHAP explanations.
- Strong engineering for deployment, but no benchmarks.

## Honourable mentions
- **Zelda-style game:** every in-game prediction is a live TabPFN call.
- **EUDR Ghana land cover:** beats RF, XGBoost, logistic regression and kNN with 25% lower log loss, plus calibration plots and a live map.

## What the judges reward (our reading)
1. TabPFN used where nobody expects it, framed as "everything is a table".
2. Several 3.5 features shown in one story: large cached context, full distributions and quantiles, text columns, Fast vs Plus.
3. **Held-out generalization tested against fair, strong baselines.**
4. Honest limitations, with any selection disclosed.
5. Polish: a hero GIF and video, a one-command demo, replay with no key needed, tests.
6. Real-world usefulness.
