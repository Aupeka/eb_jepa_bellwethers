

# Hack The World(s)

Azéma Tanguy, Kheng Augustin, Le Pendeven Théo

June 19, 2026

![Logo of MINESnancy ARTEM, featuring a stylized fan-like graphic above the text 'MINESnancy' and 'ARTEM'.](64662465bba247703fdec49c8f3309f9_img.jpg)

Logo of MINESnancy ARTEM, featuring a stylized fan-like graphic above the text 'MINESnancy' and 'ARTEM'.

![Logo of ESIEE PARIS, featuring a stylized circular graphic with four dots above the text 'ESIEE' and 'PARIS'.](5fb340ad68b0c71df0b56698b137e35b_img.jpg)

Logo of ESIEE PARIS, featuring a stylized circular graphic with four dots above the text 'ESIEE' and 'PARIS'.

![Logo of PR[AI]RIE Paris School, featuring the text 'PR[AI]RIE' in blue with 'Paris' and 'School' in black below it.](390120de4fe440c42fea8154fcaad334_img.jpg)

Logo of PR[AI]RIE Paris School, featuring the text 'PR[AI]RIE' in blue with 'Paris' and 'School' in black below it.

EN PARTENARIAT AVEC

![Logo of acadî, featuring the text 'acadî' in black and 'Réinventons le progrès' in blue.](4f4b52340aaccb1bcf733468dca9ee03_img.jpg)

Logo of acadî, featuring the text 'acadî' in black and 'Réinventons le progrès' in blue.

## Agenda

- 1 What You Have & What We Expect
- 2 Presentation Requirements
- 3 Two Rooms Example
- 4 Timeline & Judging

## Data & Project Tracks

Two kinds of data: the **datasets we provide** or **data you generate yourself**.  
Pick a track below — or use one as a springboard for your own idea.

### Group A — A new modality

- 1 Neuro - EEG
- 2 Audio / speech representation
- 3 Wearable biosignals (ECG / IMU / PPG)
- 4 Physical fields — *The Well*
- 5 Weather forecasting
- 6 Financial time series
- 7 Anomaly & predictive maintenance
- 8 3D point clouds / LiDAR

### Group B — Vision & robotics

- 9 Learned cost / value for planning
- 10 Hierarchical / multi-timescale world model
- 11 Stress-test under factors of variation
- 12 Does intuitive physics emerge?

## The Model — A JEPA Baseline

![Diagram of the JEPA Baseline model architecture.](2fa4a1bf91d0f34e87c689fbc1211fe3_img.jpg)

The diagram illustrates the JEPA Baseline model architecture, showing the flow of information and the calculation of a loss function.

**Top Track (Prediction):**

- Observation  $O_t$**  (grey box) is input to the **Encoder** (green box).
- The **Encoder** outputs a representation  $z_t$  (green arrow) to the **Predictor** (green box).
- The **Action  $a_t$**  (grey box) is also input to the **Predictor**.
- The **Predictor** outputs a **Predicted  $\hat{z}_{t+1}$**  (grey box).

**Bottom Track (Target):**

- Observation  $O_{t+1}$**  (grey box) is input to the **Target Encoder** (green box).
- The **Target Encoder** outputs a **Target  $z_{t+1}$**  (grey box).

**Loss Calculation:**

- A dashed green arrow connects the **Predicted  $\hat{z}_{t+1}$**  box to the **Target  $z_{t+1}$**  box.
- Along this arrow, the text **prediction loss + regularisation** is written in green.

Diagram of the JEPA Baseline model architecture.

### What we provide

A working **JEPA** that predicts *in representation space*, with regularisation losses to prevent collapse.  
Three tracks: **image**, **video**, **action-conditioned video**.

#### Pipeline — one engine: swappable modules

*One generic, config-driven JEPA engine in `eb_jepa/`; each example just wires swappable core modules.*

#### Core engine — `eb_jepa/`

- `jepa.py` — JEPA/JEPAProbe, unified `unroll()`
- `architectures.py` — encoders / predictors (ResNet, Impala, GRU)
- `losses.py` — predictive JEPA & anti-collapse (VICReg / VC / IDM)
- `planning.py` — CEM / MPPI planners & objectives

#### Around it — data & assembly

- `datasets/<env>/` — loaders, dispatched by `env_name`
- `examples/<x>/` — thin scripts: `main.py` · `eval.py` · `config.yaml` · README
- `launch_sbatch.py` — one launcher (`-example`, `submitit`)
- `tests/` lock the swappable-module contracts

### Pipeline — examples to reuse, tracks to complete, and above all: explore

*Reuse a complete example, complete a track baseline by filling a tiny #TODO surface — then explore.*

#### Reuse — complete examples

- `image_jepa` — CIFAR-10 SSL, linear probe ~91%
- `video_jepa` — Moving-MNIST latent prediction
- `ac_video_jepa/two_rooms` — world model + planning (~97%)
- *explore: **the maze** —  $A^*$ -free hierarchical navigation*

#### Complete — 6 tracks (baseline + #TODO)

- `audio` · `EEG` · `point cloud` · `FinTime` · `LTSF` · `Gray-Scott`
- **provided & runnable:** loader + augment + train loop + probe harness
- **you fill 3 #TODO stubs:** `build_encoder` · `build_ssl/build_jepa` · `probe`

*“Start from a baseline, then make it yours”: runnable harness → fill the stubs → your baseline → explore.*

## What We Expect From You

Start from a baseline example, then make it yours.

## 01 A working model

Train a JEPA world model on the data of your choice.

- Keep it reproducible (`config.yaml`)
- Save at least one stable checkpoint

## 02 10-minute presentation

Walk us through your approach, experiments, and findings.

- Explain your architectural choices
- Show what the model learned

### The Gold Standard

*What makes a great submission?*

- **Clear problem framing**
- **Quantitative results** vs. baseline
- **A genuine insight** or finding
- **Honest discussion** of limits
- **At least one ablation study**

![A dashed teal circle containing a teal checkmark, indicating a successful or approved status.](b241be04490fd12c763d098c5213e7c2_img.jpg)

A dashed teal circle containing a teal checkmark, indicating a successful or approved status.

### What Your Presentation Should Cover

Your 10-minute presentation should address the points below.

**Data** → **Architecture** → **Training** → **Inference** → **Evaluation**

(plus optional **Bonuses** to go further)

*Explain the data you trained on.*

#### 1 Where is it from?

- **Generated** by you
- **Chosen** from the provided datasets
- **Found elsewhere** (cite the source)

*Modality, size, difficulty?*

#### 2 What does it look like?

Show it — stats, samples, a projection (PCA):

![PCA scatter plot showing two clusters of data points.](e8ff6e66c77a8e96203c9f8db8f0986f_img.jpg)

A PCA scatter plot with two axes, PC1 (horizontal) and PC2 (vertical). The plot shows two distinct clusters of data points. The first cluster, consisting of 5 green points, is located in the upper-left quadrant. The second cluster, consisting of 5 dark blue points, is located in the lower-right quadrant. The axes intersect at the origin, and arrows indicate the positive directions for PC1 and PC2.

PCA scatter plot showing two clusters of data points.

#### 3 How did you prepare it?

- Normalisation
- Encoding / decoding target
- Data augmentation?

*Any preprocessing worth noting?*

*Describe your model and how it was trained.*

### The model

- **Unchanged?** A quick JEPA recap
- **Modified?** Why, what, intuition, papers
- Number of parameters & modules

#### Loss & training dynamics

- Which (sub)losses? how balanced?
- Representation collapse — observed?
- Regularisation & stability

![A line graph showing a training curve. The vertical axis is labeled 'loss' and the horizontal axis is labeled 'steps'. A teal curve starts at a high value on the y-axis and decreases exponentially towards the x-axis, leveling off as it approaches the right. Below the graph, the text 'training curve' is written.](33228b4227fa57e1477b27b9e07483e6_img.jpg)

training curve

A line graph showing a training curve. The vertical axis is labeled 'loss' and the horizontal axis is labeled 'steps'. A teal curve starts at a high value on the y-axis and decreases exponentially towards the x-axis, leveling off as it approaches the right. Below the graph, the text 'training curve' is written.

*How did you train, and how did you iterate efficiently?*

#### Setup

- Batch size, optimiser
- Learning-rate scheduler
- Number of epochs

##### Comparing without full training

- A **proxy metric** to rank runs early
  - A strategy to compare *without* a full run
- e.g. scaling laws, ablations, ...*

*How do you use the trained model?*

#### Inference strategy

- Method → Mode 1 / Mode 2, MCTS, actor, ...?
- Performance vs. inference time
- Any inference tricks?

#### Performance vs. time

![A line graph showing performance (perf.) on the y-axis versus time on the x-axis. The curve starts at the origin and increases at a decreasing rate, approaching a horizontal asymptote.](602ada2a012ff3cc38d91de2eec5b450_img.jpg)

A line graph with 'perf.' on the vertical y-axis and 'time' on the horizontal x-axis. A teal-colored curve starts at the origin (0,0) and rises steeply at first, then gradually levels off as it moves to the right, representing a diminishing rate of performance improvement over time.

A line graph showing performance (perf.) on the y-axis versus time on the x-axis. The curve starts at the origin and increases at a decreasing rate, approaching a horizontal asymptote.

*more compute, better rollout?*

*How good is it, and how do you know?*

##### Robustness

- Ablations
- Scaling laws
- Seed & training stability

#### Performance

- Dataset performance
- Compare to a baseline

#### Reach

- Application to real-life situations
- A step towards AGI?

*Optional — ways to stand out.*

### Rewarded

- High performance
- Ablations Hyperparameter tuning
- Extensive training-stability analysis — esp. visualising representation collapse

### Going further

- Handmade (non-synthetic) dataset
- Method that scales to other datasets of the field (e.g. time series)
- Hierarchical world models

# Two Rooms Example<sup>1</sup>

HACK THE WORLD · WORLD-MODELS TRACK

## Two-Rooms World Model

An action-conditioned JEPA for goal-reaching navigation

Team Organizer · June 2024 · example submission

Color code in this deck: red = a possible improvement gold = would be highly valued

BEFORE WE START

### How to read this deck

- It's a structure.** A clear, section-by-section way to present the results of your study.
- It's a source of ideas.** The red & gold notes suggest ways to enrich your analysis and presentation.

**✘ WHAT NOT TO DO**

Don't try to implement everything suggested here. The core of your work should be **your own research** — searching for new regularizations, borrowing methods from other fields, trying fresh ideas. These notes are only about **presenting good results excellently** — moving a good submission to an outstanding one.

OVERVIEW · THE TASK

### Goal-reaching in Two Rooms

- A dot navigates **two rooms joined by a door** — the DotWall env.
- Inputs:** pixel observations + actions.
- Learn** a world model whose latent predicts future states.
- Then plan** actions to reach a goal position.

**💡 COULD HELP** Framing the whole submission as a **scaling study** rather than a single run would be far more convincing than one number.

![Diagram of two rooms with a dot starting in one and a goal in the other.](d0a8dd8c15147e80d4923648d6561693_img.jpg)

A single two\_rooms episode: start → goal across the door.

OVERVIEW · OUR SETUP

### The example pipeline

Example submission = baseline EB-JEPA, unchanged, trained on two\_rooms via the ac\_video\_jepa recipe.

- 1 - Data generation
- 2 - JEPA training
- 3 - Evaluation

![Flowchart of the EB-JEPA architecture showing data input, representation, optimizer, cost, and planning.](f79813ca680f1c1e65e52cebf84fb57b_img.jpg)

Base EB-JEPA: action-conditioned training (top) & planning unroll (bottom).

DATA · 1/2

### Dataset selection

**Generated (synthetic).** `two_rooms` trajectories produced on-the-fly.

- Methodology:** procedural wall/door layout + random-action rollouts
- Modality:** `obs` GPU pipeline (double-buffered)
- Shape:** `obs [B, 3, T, 64, 64]`, T=17, random policy
- Compute:** ~0 storage, generated per-batch on device

**💡 COULD HELP** Finding the most revealing **visualization** would help a lot — e.g. a PCA / t-SNE of the latents colored by true XY.

**★ HIGHLY VALUED** **Offline memmaps** would enable reproducible **scaling-law** sweeps over dataset size.

![Visualizations of sampled rollouts and mean occupancy heatmap.](8977a85d40ca37cefb325af04f71e6f1_img.jpg)

Our viz: sampled rollouts + mean occupancy heatmap.

**💡 COULD HELP** Reporting difficulty knobs (door width, asymmetry) and building an **easy→hard split** would strengthen the story.

DATA · 2/2

### Preprocessing & target framing

**Normalization:** per-channel mean/std via the `two_rooms` normalizer.

**Encode target (fwd):** ground-truth frames → encoder → latent; the JEPA loss lives in this latent space.

**Decode (bwd):** frozen JEPAProbe MLP maps latent → XY position for measurable metrics.

**Augmentation:** none in baseline.

![Diagram of the preprocessing and target framing pipeline.](b6b05da4cffcded7a8e6b415a9365cf0_img.jpg)

**💡 COULD HELP** Adding **augmentation (action noise, random crops, color jitter)** would be a cheap robustness win and an easy ablation.

**💡 COULD HELP** Justifying the target — why XY? — by showing the probe is **linear-decodable** would double as a collapse sanity check.

ARCHITECTURE & LOSS · 1/3

### Model architecture

**Unchanged baseline JEPA** (recap):

- Encoder:** `ImpalaEncoder` → latent `[B, C, T, H, W]`
- Action enc:** MLP
- Predictor:** `RNNPredictor`, autoregressive, context window = 1
- Regularizer:** `VC_IDM_Sim_Regularizer`
- PredCost:** `SquareLossSeq`

~ 2.1 M params

**💡 COULD HELP** Swapping `RNNPredictor` → a **Transformer / diffusion** predictor (with papers + Δparams) could lift capacity.

**★ HIGHLY VALUED** A **hierarchical world model** — multi-timescale predictor (slow over rooms, fast over steps) — would be highly valued.

**★ HIGHLY VALUED** **Generalizing** the architecture to another modality (e.g. time-series) would really stand out.

ARCHITECTURE & LOSS · 2/3

### Loss formulation & collapse

$Total = PredCost + \lambda Regularizer$ , where the regularizer composes:

- Variance + Covariance** (anti-collapse)
- Temporal similarity**
- Inverse-dynamics (IDM)**

Baseline coeffs: `cov=1.0, var=1.0, sim=1.0, idm=1.0` (defaults).

Collapse: monitored via embedding std; held up by the variance term.

**💡 COULD HELP** A **quantitative collapse metric** (effective rank of the covariance, not just std) would make the claim solid.

**★ HIGHLY VALUED** An extensive **visual collapse analysis** — covariance heatmaps across epochs — is highly valued.

**💡 COULD HELP** Sweeping each **sub-loss coefficient** against probe error would show the balance matters.

ARCHITECTURE & LOSS · 3/3

### Training dynamics

![Training curves showing prediction cost and regularizer sub-terms over time.](8a25090502954de955851ad94c068f3d_img.jpg)

Total + prediction cost and each regularizer sub-term tracked separately — all converge smoothly, no divergence.

**💡 COULD HELP** Overlaying an **unstable run** (too-high LR) beside the smooth one would nicely illustrate how hyperparameters drive stability.

<sup>0</sup>Found it at [hackathon\\_guide/two\\_rooms\\_example\\_slides.pdf](#)

Azéma, Kheng, Le Pendeven
What You Have
Requirements
Two Rooms
Logistics
15 / 17

### Schedule

17:30 Code submission

18:00 Slides submission & pre-jury

19:00 Final jury (*qualified teams*)

![Timeline diagram showing three time points: 17:30, 18:00, and 19:00.](fc857414626a8d94d132e12d9afe52a4_img.jpg)

A horizontal timeline with three points marked by dots. The first dot is teal and labeled '17:30'. The second dot is teal and labeled '18:00'. The third dot is dark blue and labeled '19:00'. A thin grey line connects the first two dots, and a thin dark blue line connects the second and third dots.

Timeline diagram showing three time points: 17:30, 18:00, and 19:00.

## Judging Criteria

### - **Technical quality**

*Robustness, code cleanliness & architecture*

### - **Results**

*Performance, metrics & meaningful comparisons*

### - **Presentation**

*Clarity, storytelling & visual aids*

### - **Originality**

*Ablations, novel insights & going the extra mile*

# Now go build your world.

*Good luck, and have fun!*