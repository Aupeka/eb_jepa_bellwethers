

HACK THE WORLD · WORLD-MODELS TRACK

# Two-Rooms World Model

An action-conditioned JEPA for goal-reaching navigation

Team **Organizer** · June 2026 · example submission

Color code in this deck — red = a possible improvement gold = an exploration possibility

BEFORE WE START

## How to read this deck

▵ **It's a structure.** A clear, section-by-section way to **present the results** of your study.

💡 **It's a source of ideas.** The red & gold notes suggest ways to **enrich your analysis and presentation.**

### X WHAT NOT TO DO

Don't try to implement everything suggested here. The core of your work should be **your own research** — searching for new regularizations, borrowing methods from other fields, trying fresh ideas. These notes are only about **presenting good results excellently** — moving a good submission to an outstanding one.

## Goal-reaching in Two Rooms

- A dot navigates **two rooms joined by a door** — the DotWall env.
- **Inputs:** pixel observations + actions.
- **Learn** a world model whose latent predicts future states.
- **Then plan** actions to reach a goal position.

◇ **COULD HELP** Framing the whole submission as a **scaling study** rather than a single run would be far more convincing than one number.

![Diagram of the DotWall environment showing two rooms separated by a vertical wall with a door at the top. A black dot labeled 'start' is in the left room, and a green square labeled 'goal' is in the right room.](48f188337e3ba41df38fab9ac0afb1bd_img.jpg)

A diagram of the DotWall environment. It consists of a square arena divided into two rooms by a vertical black wall. A small black dot, labeled 'start', is positioned in the left room. A small green square, labeled 'goal', is positioned in the right room. The two rooms are connected by a horizontal opening at the top of the vertical wall, representing a door. The arena is bounded by black lines on the top, bottom, and sides.

Diagram of the DotWall environment showing two rooms separated by a vertical wall with a door at the top. A black dot labeled 'start' is in the left room, and a green square labeled 'goal' is in the right room.

A single two\_rooms episode: start → goal across the door.

## The example pipeline

**Example submission = baseline EB-JEPA, unchanged,**  
trained on two\_rooms via the ac\_video\_jepa recipe.

### 1 · Data generation

two\_rooms rollouts

![Downward arrow indicating flow from data generation to training.](af8b95b7fc833cebe89ba6c8ed839984_img.jpg)

Downward arrow indicating flow from data generation to training.

### 2 · JEPA training

encoder + predictor + action module

![Downward arrow indicating flow from training to evaluation.](ab5392ddbf4af970d4ef7bff2e4df925_img.jpg)

Downward arrow indicating flow from training to evaluation.

### 3 · Evaluation

actor plans to the goal

![Diagram illustrating the Base EB-JEPA pipeline, showing action-conditioned training (top) and planning unroll (bottom).](37a6ab1d23efb9dc00cfae09d353b1da_img.jpg)

The diagram illustrates the Base EB-JEPA pipeline, divided into two main sections: Action-Conditioned Video (top) and Planning (bottom).

**Action-Conditioned Video (Top):** This section shows the training process. It features two parallel sequences of inputs  $x_t$  and  $x_{t+1}$  passing through an encoder  $f_\theta$  to produce representations  $z_t$  and  $z_{t+1}$ . The representation  $z_t$  is also influenced by an action  $a_t$ . These representations are fed into a predictor  $g_\phi$  to generate a prediction  $\hat{z}_{t+1}$ . A cost module  $C$  takes  $z_{t+1}$  and  $\hat{z}_{t+1}$  as inputs to calculate the cost.

**Planning (Bottom):** This section shows the planning process. It starts with an initial state  $x_t$  passing through  $f_\theta$  to  $z_t$ , which is then processed by  $g_\phi$  (receiving action  $a_t$ ) to produce  $\hat{z}_T$ . This is followed by a sequence of representations  $z_g$  and  $\hat{z}_g$ , which are processed by  $f_\theta$  and  $g_\phi$  (receiving action  $a_{T-1}$ ) to reach the goal state  $x_g$ . A cost module  $C$  is also shown.

**Legend:**

- $x$ : Data input (grey circle)
- $z$ : Representation (grey rectangle)
- $a$ : Optimized (orange circle)
- $C$ : Cost (orange rectangle)

Diagram illustrating the Base EB-JEPA pipeline, showing action-conditioned training (top) and planning unroll (bottom).

Base EB-JEPA: action-conditioned training (top) & planning unroll (bottom).

### Dataset selection

**Generated (synthetic).** two\_rooms trajectories produced on-the-fly.

- **Methodology:** procedural wall/door layout + random-action rollouts
- **Modality:** `online` GPU `stream` pipeline (double-buffered)
- **Shape:** obs  $[B, 3, T, 64, 64]$ ,  $T=17$ , random policy
- **Compute:**  $\sim 0$  storage, generated per-batch on device

◇ **COULD HELP** Finding the most revealing **visualization** would help a lot — e.g. a PCA / t-SNE of the latents colored by true XY.

★ **EXPLORATION** Freezing a fixed dataset to disk (**offline memmaps**) → identical data across runs, so you can chart how performance grows with **more training data**.

![Two side-by-side visualizations: 'Sampled rollouts (random policy)' and 'Mean occupancy heatmap'.](6f1efa91fb9b476380af7a35db4f14bf_img.jpg)

The figure consists of two side-by-side visualizations. The left visualization, titled 'Sampled rollouts (random policy)', shows a top-down view of a two-room environment with a central doorway. Multiple colored lines (red, blue, green, orange) represent the paths of agents starting from different points and moving randomly. The right visualization, titled 'Mean occupancy heatmap', shows the same environment with a color-coded heatmap representing the frequency of visits. A color bar on the right indicates 'log visits' ranging from 0.0 (dark purple) to 3.0 (yellow). The heatmap shows higher visitation (yellow/orange) in the open areas of both rooms and lower visitation (purple) near the walls and the doorway.

Two side-by-side visualizations: 'Sampled rollouts (random policy)' and 'Mean occupancy heatmap'.

Our viz: sampled rollouts + mean occupancy heatmap.

◇ **COULD HELP** Reporting difficulty knobs (door width, asymmetry) and building an **easy→hard split** would strengthen the story.

### Preprocessing & target framing

**Normalization:** per-channel mean/std via the two\_rooms normalizer.

**Encode target (fwd):** ground-truth frames  $\rightarrow$  encoder  $\rightarrow$  latent; the JEPA loss lives in this latent space.

**Decode (bwd):** frozen JEPAProbe MLP maps latent  $\rightarrow$  XY position for measurable metrics.

**Augmentation:** *none* in baseline.

![Bomb icon indicating a syntax error.](35afbfc3c4a5c0fe01e91ba536605e09_img.jpg)

A dark red bomb icon with a lit fuse and small sparks, used to highlight a syntax error in the text.

Bomb icon indicating a syntax error.

Syntax error in text  
mermaid version 11.15.0

◇ **COULD HELP** Adding augmentation (**action noise, random crops, color jitter**) would be a cheap robustness win and an easy ablation.

◇ **COULD HELP** Justifying the target — why XY? — by showing the probe is **linear-decodable** would double as a collapse sanity check.

### Model architecture

Unchanged baseline JEPA (recap):

- **Encoder:** ImpalaEncoder  $\rightarrow$  latent  $[B, C, T, h, w]$
- **Action enc:** MLP
- **Predictor:** RNNPredictor, autoregressive, ctxt window = 1
- **Regularizer:** VC\_IDM\_Sim\_Regularizer
- **PredCost:** SquareLossSeq

≈ 2.1 M params

no custom modules

◇ **COULD HELP** Swapping RNNPredictor  $\rightarrow$  a **Transformer / diffusion** predictor (with papers +  $\Delta$ params) could lift capacity.

★ **EXPLORATION** A **hierarchical world model** — a multi-timescale predictor (slow over rooms, fast over steps).

### Loss formulation & collapse

**Total** = PredCost +  $\lambda$ ·Regularizer, where the regularizer composes:

- **Variance + Covariance** (anti-collapse)
- **Temporal similarity**
- **Inverse-dynamics (IDM)**

Baseline coeffs: cov=1.0, var=1.0, sim=1.0, idm=1.0 (defaults).

Collapse: monitored via embedding std; held up by the variance term.

◇ **COULD HELP** A **quantitative collapse metric** (effective rank of the covariance, not just std) would make the claim solid.

★ **EXPLORATION** An extensive **visual collapse analysis** — e.g. covariance heatmaps across epochs.

◇ **COULD HELP** Sweeping each **sub-loss coefficient** against probe error would show the balance matters.

### Training dynamics

![A 3x3 grid of line charts showing training dynamics across 25,000 steps for nine different metrics. Each chart displays multiple runs with shaded confidence intervals.](91be14371a97fb5ce9eeb29ae18d07c3_img.jpg)

The figure consists of nine subplots arranged in a 3x3 grid, each showing a different training metric over 25,000 steps. Each plot includes a title, a legend with three entries (ojFalse\_simtprojFalse\_1sttFalse\_rolldstc32\_henc3, jFalse\_simtprojFalse\_1sttFalse\_rolldstc32\_henc3, and oFalse\_simtprojFalse\_1sttFalse\_rolldstc32\_henc3), and a y-axis label. The x-axis for all plots is 'Step' ranging from 0 to 2.5k.

- success\_rate**: Y-axis 0.2 to 1.0. Shows an upward trend from ~0.4 to ~0.9.
- train/pred\_loss**: Y-axis 0.02 to 0.2. Shows a downward trend from ~0.15 to ~0.05.
- train/reg\_loss\_unweight**: Y-axis 0.2 to 0.6. Shows a downward trend from ~0.5 to ~0.2.
- train/reg/cov\_loss**: Y-axis 0.005 to 0.05. Shows a downward trend from ~0.04 to ~0.01.
- train/reg/idm\_loss**: Y-axis 0.1 to 0.5. Shows a downward trend from ~0.4 to ~0.1.
- train/reg/sim\_loss\_t**: Y-axis 0.002 to 0.05. Shows a downward trend from ~0.04 to ~0.01.
- train/reg/std\_loss**: Y-axis 0.001 to 0.01. Shows a downward trend from ~0.008 to ~0.002.
- val\_rollout/mean\_pos\_mse/1**: Y-axis 0.001 to 0.01. Shows a downward trend from ~0.008 to ~0.002.
- val\_rollout/mean\_pos\_mse/6**: Y-axis 0.003 to 0.04. Shows a downward trend from ~0.03 to ~0.005.

A 3x3 grid of line charts showing training dynamics across 25,000 steps for nine different metrics. Each chart displays multiple runs with shaded confidence intervals.

A real training sweep across runs: **success rate** rises while the **prediction & regularizer losses** and **validation position error** all fall — smooth, stable convergence.

◇ **COULD HELP** Overlaying an **unstable run** (too-high LR) beside the smooth one would nicely illustrate how hyperparameters drive stability.

### Configuration & fast iteration

| Knob        | Baseline         |
|-------------|------------------|
| Batch size  | 256              |
| Optimizer   | AdamW            |
| LR schedule | CosineWithWarmup |
| Peak LR     | 3e-4             |
| Epochs      | 50               |
| Seeds       | 1 / 1000 / 10000 |

**Efficient iteration:** rank runs by **probe XY-MSE** at epoch 5 before committing to full training.

◇ **COULD HELP** Showing the **proxy metric correlates** with final performance (early-vs-final scatter) would make this rigorous.

★ **EXPLORATION** **Comprehensive hyperparameter tuning** and **scaling laws** (loss vs params / data / compute).

## INFERENCE

### Planning strategy

**Mode 2 (planning):** GCAGENT + CEMPLANNER with the ReprTargetDistMPCObjective — minimize latent distance to the goal's encoding via MPC.

![Flowchart of the planning strategy](5a4e62bead259c258d069fd3663ea670_img.jpg)

```
graph LR; State[State] --> Encode[Encode]; Encode --> CEM[CEM: sample actions]; CEM --> Unroll[Unroll model]; CEM --> Best[Best action]; Unroll --> Latent[Latent dist to goal]; Best --> Latent; Latent --> CEM;
```

The diagram illustrates the planning strategy flow. It starts with a 'State' box, which points to an 'Encode' box. The 'Encode' box points to a 'CEM: sample actions' box. From 'CEM: sample actions', there are two paths: one to 'Unroll model' and another to 'Best action'. Both 'Unroll model' and 'Best action' point to a 'Latent dist to goal' box. Finally, the 'Latent dist to goal' box points back to the 'CEM: sample actions' box, forming a feedback loop.

Flowchart of the planning strategy

| Method          | Success | ms/step |
|-----------------|---------|---------|
| Mode 1 reactive | 61%     | 2       |
| CEM (baseline)  | 84%     | 45      |

◇ **COULD HELP** Comparing **MPPI vs CEM** and sweeping horizon & samples would give a **perf-vs-compute** Pareto curve.

★ **EXPLORATION** A **custom inference trick** — distilling the planner into a fast amortized policy, or **MCTS / actor-critic**.

## EVALUATION

### Robustness & performance

**Ablation:** remove the IDM term → probe XY-MSE +38%, planning success 84%→72%. IDM matters.

**Seed stability:** success 84%  $\pm$  3% across seeds 1/1000/10000.

**vs baseline:** matches reference numbers (this is the baseline).

◇ **COULD HELP** Running **multiple ablations** (sub-losses, predictor, context window) with **seed error bands** would deepen the robustness story.

◇ **COULD HELP** A **generalization** test — train on some wall layouts, evaluate on unseen ones — would be revealing.

★ **EXPLORATION** **Broader impact** — links to robot navigation and world-models as a step toward AGI.

## Thank you

This example covers the basics — each **red note** in the deck is a concrete next step that would enhance the presentation and the analysis of your results.

Two-Rooms JEPA · an example that knows how to grow up

Team Organizer — questions welcome