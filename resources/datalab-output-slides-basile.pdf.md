

# EB-JEPA — A Lightweight Library for Energy-Based

## JEPA

JEPA, from image representation learning to world models · For education & research · Single GPU in hours

![Diagram illustrating the EB-JEPA architecture for different tasks: Image, Video, Action-Conditioned Video, and Planning.](68d50e85fb8de4fae0e0eafaf20e63c0_img.jpg)

The diagram illustrates the EB-JEPA architecture for four different tasks, each within a light blue box. A legend on the right defines the symbols: a grey circle for 'Data input', a grey rectangle for 'Representation', an orange circle for 'Optimized', and an orange rectangle for 'Cost'.

- (a) Image: Two data inputs  $x$  and  $x'$  are processed by functions  $f_\theta$  to produce representations  $z$  and  $z'$ . These are then compared using a cost function  $C$ .
- (b) Video: A sequence of data inputs  $x_t$  and  $x_{t+1}$  are processed by  $f_\theta$  to produce representations  $z_t$  and  $z_{t+1}$ .  $z_t$  is also processed by a generator  $g_\phi$  to produce a prediction  $\hat{z}_{t+1}$ , which is then compared with  $z_{t+1}$  using cost  $C$ .
- (c) Action-Conditioned Video: Similar to (b), but an action input  $a_t$  is also provided to the generator  $g_\phi$ .
- (d) Planning: A sequence of data inputs  $x_t$  and  $x_g$  are processed by  $f_\theta$  to produce representations  $z_t$  and  $z_g$ .  $z_t$  is processed by a generator  $g_\phi$  to produce a prediction  $\hat{z}_T$ , which is compared with  $z_g$  using cost  $C$ . An action input  $a_t$  is also provided to the generator  $g_\phi$ .

Legend:

- $x$  (grey circle): Data input
- $z$  (grey rectangle): Representation
- $a$  (orange circle): Optimized
- $C$  (orange rectangle): Cost

Diagram illustrating the EB-JEPA architecture for different tasks: Image, Video, Action-Conditioned Video, and Planning.

Basile Terver<sup>12</sup> · Randall Balestriero<sup>1</sup> · Megi Dervishi<sup>1</sup> · David Fan<sup>1</sup> · Quentin Garrido<sup>1</sup> · Tushar Nagarajan<sup>1</sup> · Koustuv Sinha<sup>1</sup> · Wancong Zhang<sup>1</sup> · Mike Rabbat<sup>1</sup> · Yann LeCun<sup>13†</sup> · Amir Bar<sup>1†</sup>

<sup>1</sup>Meta FAIR · <sup>2</sup>INRIA · <sup>3</sup>NYU · <sup>†</sup>Equal contribution · ICLR 2026 · 2nd Workshop on World Models

![QR code linking to the EB-JEPA repository on GitHub.](0538daaa5583c23e17db3a12f2281a55_img.jpg)

QR code linking to the EB-JEPA repository on GitHub.

## Key concepts: the JEPA recipe

A JEPA predicts in representation space and regularizes so the representation stays informative.

### Encoder

$$z = f_{\theta}(x)$$

Maps observation to a compact embedding that captures semantics, not pixels.

### Target encoder

$$z' = f_{\bar{\theta}}(x')$$

A stop-gradient / EMA copy of the encoder that produces stable prediction targets.

### Collapse

$$z \equiv \text{const}$$

A constant embedding makes  $E = 0$  trivially but encodes nothing. The core failure mode.

### Predictor

$$\hat{z} = g_{\phi}(z, a)$$

Forecasts target embedding from context, optionally conditioned on action  $a$ .

### Energy

$$E = \|\hat{z} - z'\|^2$$

Low when the prediction matches the target. Trained by gradient descent — this

### Regularizer

$$\mathcal{R}(z)$$

Keeps embeddings spread out & decorrelated (VICReg / SIGReg) so collapse is avoided.

![Diagram of the JEPA architecture showing the flow from inputs x and y through encoders, a predictor, and a discriminator, with associated loss functions and objectives.](662536cd1f5fc0a0b339e73307f077d7_img.jpg)

The diagram illustrates the JEPA architecture. It shows two input observations,  $x$  and  $y$ , each passing through an encoder,  $\text{Enc}(x)$  and  $\text{Enc}(y)$ , to produce embeddings  $s_x$  and  $s_y$ . The embedding  $s_x$  is used by the Predictor,  $\text{Pred}(s_x, z)$ , along with a latent variable  $z$  to produce a predicted embedding  $\tilde{s}_y$ . The Predictor is trained to minimize the prediction error,  $D(s_y, \tilde{s}_y)$ . The latent variable  $z$  is also used by a Regularizer,  $R(z)$ , to minimize information content. The embeddings  $s_x$  and  $s_y$  are also used to calculate the negative information content,  $-I(s_x)$  and  $-I(s_y)$ , which are maximized to keep the representations informative. The diagram uses color coding: green for the Predictor, red for the Discriminator and information content terms, and purple for the encoders.

Diagram of the JEPA architecture showing the flow from inputs x and y through encoders, a predictor, and a discriminator, with associated loss functions and objectives.

## Why JEPA? And why a lightweight library?

### The pixel prediction problem

Video-generation world models reconstruct **every pixel**, including task-irrelevant detail — forcing huge compute budgets to learn mostly decorative features.

JEPAs instead predict in a **learned representation space**, concentrating capacity on semantically useful structure.

### The accessibility gap

- Production JEPA code (I-JEPA, V-JEPA) is tuned for large-scale runs
- World-model variants (DINO-WM, JEPA-WMs) assume frozen backbones or are focused on world modeling only (stable-worldmodel)
- **Barrier to entry** is high for researchers & students

**EB-JEPA:** three self-contained, single-GPU examples — image · video · action-conditioned world model — that make the energy-based JEPA recipe accessible.

### A unified energy-based view

All three settings share one energy function — complexity grows as we move from static images to video to action-conditioned dynamics.

$$E(x, x', a) = \mathcal{L}_{\text{pred}}(g_\phi(f_\theta(x), q_\omega(a)), f_\theta(x'))$$

**(a) Image-JEPA — view invariance**

$g_\phi = \text{Id}$ , no actions.

$$E = \|z - z'\|^2 + \lambda \mathcal{R}(z)$$

**(b) Video-JEPA — temporal prediction**

$g_\phi$  predicts  $z_{t+1}$  from  $z_{t-v:t}$ .

No actions.

**(c) AC-Video-JEPA — world model**

$g_\phi$  conditioned on  $u_t = q_\omega(a_t)$ .

Supports planning.

## Preventing collapse: two regularizer families

The JEPA objective has a trivial minimum — constant  $z$ . Regularization is what makes these models work.

### VICReg

$$\mathcal{R}_{\text{VICReg}} = \alpha \mathcal{L}_{\text{var}} + \beta \mathcal{L}_{\text{cov}}$$

$$\mathcal{L}_{\text{var}} = \frac{1}{d} \sum_j \max(0, \gamma - \sqrt{\text{Var}(z_j)} + \epsilon)$$

$$\mathcal{L}_{\text{cov}} = \frac{1}{d} \sum_{i \neq j} [\text{Cov}(z)]_{ij}^2$$

- One pair of hyperparameters ( $\alpha$ ,  $\beta$ ); in practice cov-weight  $\approx 10 \times$  var.

### SIGReg (LeJEPA)

$$\mathcal{R}_{\text{SIGReg}}(Z) = \frac{1}{P} \sum_{p=1}^P \mathcal{G}(Z \xi_p)$$

- Enforces embeddings  $\sim N(0, I)$  via 1-D Gaussianity tests
- **One** hyperparameter  $\lambda$ , linear cost
- Provably optimal distribution

For world models, two more terms — both critical under randomized environments:

$$\mathcal{L}_{\text{sim}} \text{ (smooth trajectories)} \quad \mathcal{L}_{\text{IDM}} \text{ (inverse dynamics)}$$

#### Image-JEPA on CIFAR-10

**Setup** — ResNet-18, 300 epochs, linear probe on frozen features.

![Diagram of the Image-JEPA architecture. Two input images, x1 and x2, are sampled from a distribution t ~ T. Each image is processed by an encoder E_theta to produce embeddings s1 and s2. These embeddings are then passed through a projector F_phi to produce representations z1 and z2. A cost function R is applied to z1 and z2, and a loss L2 is calculated between them. A legend indicates: E_theta is the encoder, F_phi is the projector, a square box is the cost function, and t ~ T is augmentation.](a7d78d22e465dea388b31d0739f9d0cd_img.jpg)

Diagram illustrating the Image-JEPA architecture. Two input images,  $x_1$  and  $x_2$ , are sampled from a distribution  $t \sim T$ . Each image is processed by an encoder  $E_\theta$  to produce embeddings  $s_1$  and  $s_2$ . These embeddings are then passed through a projector  $F_\phi$  to produce representations  $z_1$  and  $z_2$ . A cost function  $R$  is applied to  $z_1$  and  $z_2$ , and a loss  $L_2$  is calculated between them. A legend indicates:  $E_\theta$  is the encoder,  $F_\phi$  is the projector, a square box is the cost function, and  $t \sim T$  is augmentation.

Diagram of the Image-JEPA architecture. Two input images, x1 and x2, are sampled from a distribution t ~ T. Each image is processed by an encoder E\_theta to produce embeddings s1 and s2. These embeddings are then passed through a projector F\_phi to produce representations z1 and z2. A cost function R is applied to z1 and z2, and a loss L2 is calculated between them. A legend indicates: E\_theta is the encoder, F\_phi is the projector, a square box is the cost function, and t ~ T is augmentation.

![Two line plots comparing VICReg and SIGReg performance. The left plot, titled 'VICReg Performance', shows Validation Accuracy (%) vs Epoch (0 to 300). It features three lines: 'Collapsing runs' (purple, low accuracy), 'Normal runs' (blue, rising to ~80%), and 'Best run: 90.12' (red, rising to ~85%). The right plot, titled 'SIGReg Performance', shows Validation Accuracy (%) vs Epoch (0 to 300). It features three lines: 'Collapsing runs' (purple, low accuracy), 'Normal runs' (blue, rising to ~85%), and 'Best run: 91.02' (red, rising to ~88%).](46f43cb4ffd47565e7c0ca306d461435_img.jpg)

Two line plots comparing VICReg and SIGReg performance. The left plot, titled "VICReg Performance", shows Validation Accuracy (%) vs Epoch (0 to 300). It features three lines: "Collapsing runs" (purple, low accuracy), "Normal runs" (blue, rising to ~80%), and "Best run: 90.12" (red, rising to ~85%). The right plot, titled "SIGReg Performance", shows Validation Accuracy (%) vs Epoch (0 to 300). It features three lines: "Collapsing runs" (purple, low accuracy), "Normal runs" (blue, rising to ~85%), and "Best run: 91.02" (red, rising to ~88%).

Two line plots comparing VICReg and SIGReg performance. The left plot, titled 'VICReg Performance', shows Validation Accuracy (%) vs Epoch (0 to 300). It features three lines: 'Collapsing runs' (purple, low accuracy), 'Normal runs' (blue, rising to ~80%), and 'Best run: 90.12' (red, rising to ~85%). The right plot, titled 'SIGReg Performance', shows Validation Accuracy (%) vs Epoch (0 to 300). It features three lines: 'Collapsing runs' (purple, low accuracy), 'Normal runs' (blue, rising to ~85%), and 'Best run: 91.02' (red, rising to ~88%).

SIGReg is flatter across HPs; VICReg peaks higher but needs care.

| Method | Best acc. | w/o proj. | #HPs |
|--------|-----------|-----------|------|
| SIGReg | 91.0%     | -3.3 pts  | 1    |
| VICReg | 90.1%     | -2.9 pts  | 2    |

### Findings

- **Projector helps** — adds ~3 pts over direct regularization
- **Bottleneck matters** — SIGReg 2048→128 vs VICReg 2048→1024
- **SIGReg easier to tune** — single  $\lambda$ ; VICReg needs cov  $\sim 10 \times$  std
- Bad hyperparams collapse to ~10% (random)

### Video-JEPA on Moving MNIST

![Diagram of Video-JEPA architecture showing multistep rollout training loss.](1956f44611abd5c3c41049836aa78ad8_img.jpg)

The diagram illustrates the Video-JEPA architecture for Moving MNIST. It shows three parallel video sequences (represented by frames with a digit '0') being processed by an encoder  $E_\theta$  to produce states  $s_0$ ,  $s_1$ , and  $s_2$ . Each state  $s_k$  is also passed through a regularizer  $R$ . The states are then fed into a predictor  $G_\phi$  to generate predictions  $\hat{s}_1$  and  $\hat{s}_2$ . The predictions are compared with the ground truth states using an  $L_2$  cost function. The total prediction loss is defined as:

$$\mathcal{L}_{\text{pred}} = \sum_{k=1}^K \mathcal{L}_k$$

Legend:

- $G_\phi$ : predictor
- $E_\theta$ : encoder
- $\square$ : cost function\*

\*input is first projected via an MLP

Diagram of Video-JEPA architecture showing multistep rollout training loss.

- **Multistep rollout training loss** — aligns training with autoregressive inference
- Reduces exposure bias  $\rightarrow$  better long-horizon prediction

![Training dynamics plots for VC Loss, Pred Loss, and mAP over 50 epochs.](a6a8016b231533e7f34b550f4676afc6_img.jpg)

Three plots showing training dynamics over 50 epochs:

- VC Loss**: Decreases from approximately 4.0 to 0.0.
- Pred Loss**: Decreases from approximately 0.5 to 0.0.
- mAP**: Increases from approximately 0.1 to 0.6.

Training dynamics plots for VC Loss, Pred Loss, and mAP over 50 epochs.

Training dynamics (50 epochs): regularizer R, prediction loss, downstream mAP.

![AP vs Timestep plot for different rollout steps.](4279c8be6ec4ed56f4b3349be98bb426_img.jpg)

Plot of AP (Average Precision) vs Timestep for different rollout steps:

- 1 step (blue line): AP decreases from ~0.8 to ~0.15.
- 2 step (orange line): AP decreases from ~0.8 to ~0.25.
- 4 step (green line): AP decreases from ~0.8 to ~0.4.
- 8 step (red line): AP decreases from ~0.8 to ~0.45.

AP vs Timestep plot for different rollout steps.

k-step rollouts beat single-step. Pareto optimum around k=4.

### Another collapse mode: the slow-feature

$$\mathcal{L}_{\text{variance}} = \frac{1}{(T+1)D} \sum_{t=1}^{T+1} \sum_{j=1}^D \max\left(0, \gamma - \sqrt{\text{Var}(S_{t,:,j}) + \epsilon}\right) = 0 \quad \text{Var}(s_t) = \frac{1}{N-1} \sum_{i=1}^N (s_i - \bar{s})^2 = \sigma^2 \quad \text{As } s \sim \mathcal{N}(0, \sigma^2 I)$$
$$\mathcal{L}_{\text{covariance}} = \frac{1}{(T+1)(N-1)} \sum_{t=1}^{T+1} \sum_{i=1}^D \sum_{j=i+1}^D (S_t S_t^\top)_{i,j} = 0 \quad \mathcal{L}_{\text{pred}} = \frac{1}{NT} \sum_{t,i} \|\hat{z}_{t,i} - z_{t,i}\|^2 = 0$$

### The setup

$$\hat{o}_t = o_t + \alpha Z_t$$

- $o_t$  = a moving dot (the signal to learn)
- $Z_t$  = a distractor background
- **Changing noise:** resampled every frame  $\rightarrow$  unpredictable
- **Fixed noise:** constant within a sequence, random across sequences

### The zero-loss trivial solution (fixed noise)

$$g_\phi(o_t) = s, \quad f_\theta(s, a) = s$$

- the encoder grabs the fixed noise, predictor = identity
- $L_{\text{pred}} = 0$  since  $z_t = z_{t+1} = s$
- $L_{\text{var}} \approx 0$  (noise has spread across sequences)
- $L_{\text{cov}} = 0$  (noise dimensions are independent)
- $\rightarrow$  total loss = 0, yet the dot is never encoded

**Why this matters for us.** Prediction + variance + covariance can all be satisfied by **slow, predictable-but-spurious** features. Forcing the encoder to capture action-controllable content needs an extra signal — the **inverse-dynamics (IDM)** term. This is exactly why IDM is load-bearing in our AC-Video ablation: removing it collapses planning success to  **$1 \pm 1\%$** .

### Action-Conditioned Video-JEPA → planning

![Diagram illustrating the architecture of Action-Conditioned Video-JEPA for planning.](acfc53eca625d62b38aa2563efa95c3e_img.jpg)

The diagram illustrates the architecture of Action-Conditioned Video-JEPA for planning, divided into three main components: Action-Conditioned Video, Planning, and Task Definition.

**Action-Conditioned Video:** This component shows the internal structure of the video processing module. It takes data input  $x_t$  and action  $a_t$  as inputs to a function  $f_\theta$ , which produces a representation  $z_t$ . This representation is then processed by a function  $g_\phi$  to produce the next representation  $\hat{z}_{t+1}$ . A cost function  $C$  is applied to the next representation  $\hat{z}_{t+1}$  to produce the cost  $C$ .

**Planning:** This component shows the planning process. It starts with a data input  $x_t$  and action  $a_t$ , which are processed by  $f_\theta$  to produce a representation  $z_t$ . This representation is then processed by a sequence of  $g_\phi$  functions to produce a sequence of representations  $\hat{z}_T$ . The cost function  $C$  is applied to the final representation  $\hat{z}_T$  to produce the cost  $C$ . The cost is then used to produce the next action  $a_{T-1}$ , which is fed back into the planning process.

**Task Definition:** This component shows the task definition. It starts with an initial state  $x_g$  and a goal state  $x_g$ . The initial state is processed by  $f_\theta$  to produce a representation  $z_g$ , which is then processed by  $g_\phi$  to produce the next representation  $\hat{z}_g$ . The cost function  $C$  is applied to the final representation  $\hat{z}_g$  to produce the cost  $C$ .

**Legend:**

- $x$ : Data input
- $z$ : Representation
- $a$ : Optimized
- $C$ : Cost

Diagram illustrating the architecture of Action-Conditioned Video-JEPA for planning.

![Diagram illustrating the task definition and successful planning episode.](562f471e8153729557e6a4ee6343c32c_img.jpg)

The diagram illustrates the task definition and a successful planning episode. It shows two rows of images, each representing a different task definition.

**Task Definition:** The top row shows the initial state (Init) and the goal state (Goal). The bottom row shows the initial state (Init) and the goal state (Goal).

**Successful planning episode:** The left column shows a sequence of frames from a video episode, labeled "Frame 1/201". The right column shows the initial state (Init) and the goal state (Goal).

**Episode task definition:** The right column shows the initial state (Init) and the goal state (Goal).

Diagram illustrating the task definition and successful planning episode.

**Two Rooms (random wall):** non-monotonic — agent must first move **away** from the goal.

$E_{plan}$  minimized with any relevant planning optimizer — these can be gradient-based, sampling-based or hybrid

$$E_{plan}(a_{0:H}) = \sum_{t=1}^H \|f_\theta(x_g) - \hat{z}_t\|$$

### What matters: ablations

$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \alpha \mathcal{L}_{\text{var}} + \beta \mathcal{L}_{\text{cov}} + \delta \mathcal{L}_{\text{sim}} + \omega \mathcal{L}_{\text{IDM}}$$

Each regularization term pulls its weight — removing one often kills the model.

- **IDM is load-bearing** — remove it and the encoder collapses onto slow features only (spurious correlations in randomized env)
- Variance & covariance each worth ~50 pts absolute
- Temporal similarity ~35 pts
- **Planner design matters too** — summing intermediate states, not just final, encourages faster goal-reaching (+8 pts)

| Ablated term             | Success ↑ |
|--------------------------|-----------|
| None (full model)        | 97 ± 2%   |
| Variance $\alpha=0$      | 47 ± 3%   |
| Covariance $\beta=0$     | 46 ± 3%   |
| Temporal Sim. $\delta=0$ | 61 ± 2%   |
| IDM $\omega=0$           | 1 ± 1%    |

*Planner ablation (cumulative vs final-state cost)*

| Planner                 | Success ↑ |
|-------------------------|-----------|
| MPPI (cumulative)       | 97 ± 2%   |
| CEM (cumulative)        | 96 ± 2%   |
| MPPI (final-state only) | 89 ± 2%   |

3 seeds × 3 checkpoints × 20 episodes. IDM ablation yields total collapse.

## Related: frozen-encoder world models (JEPA-WMs)

### The JEPA-WM family

A frozen pretrained encoder (DINOv2/3, V-JEPA/2) + a predictor trained by a predictive loss in embedding space — no reconstruction, no reward or value heads.

**Contrast with EB-JEPA:** these baselines learn only the *dynamics*. We learn encoder *and* predictor jointly, from scratch, on a single GPU.

![Diagram illustrating the JEPA-WM architecture for Training and Planning.](d26959f4514c26ca19c3d6f00da85956_img.jpg)

The diagram is divided into two main sections: Training and Planning.

**Training:** Shows a sequence of images  $o_t, \dots, o_{t+3}$  being processed by an encoder  $E_\theta$  to produce embeddings  $z_t, z_{t+1}, z_{t+2}, z_{t+3}$ . These embeddings, along with actions  $a_{t:t+2}$ , are fed into a predictor  $P_\theta$ . The predictor outputs predicted embeddings  $\hat{z}_{t+1}, \hat{z}_{t+2}, \hat{z}_{t+3}$ . A JEPA teacher-forcing loss  $L$  is calculated between the predicted embeddings and the ground truth embeddings  $z_{t+1}, z_{t+2}, z_{t+3}$ .

**Planning:** Shows a sequence starting from a "Start image"  $o_t$  (encoded to  $z_t$ ) and a sequence of actions  $a_t, \dots, a_{t+H-1}$ . These are fed into a "Predictor unroll  $F_\theta$ " which produces a sequence of predicted embeddings  $\hat{z}_{t+1}, \dots, \hat{z}_{t+H-1}$ . This sequence is then fed into another predictor  $P_\theta$  along with action  $a_{t+H-1}$  to produce a final predicted embedding  $\hat{z}_{t+H}$ . This is compared with a goal embedding  $z_g$  (from a "Goal image"  $o_g$ ) using a "Planning cost  $L^p$ ".

Diagram illustrating the JEPA-WM architecture for Training and Planning.

JEPA-WM: predictor trained by teacher forcing (left); plan by unrolling it (right).

#### V-JEPA-2-AC

Frozen V-JEPA-2 (ViT-L). Action & proprioception encoded as tokens concatenated along the sequence (RoPE).

#### DINO-WM

Frozen DINOv2. Action & proprioception via linear layers concatenated along the embedding dimension (sincos).

## Beyond the three examples: hackathon tracks

Same engine, one #TODO surface per track — a track = data loaders + a task declaration. Single-GPU throughout.

| Track                | SSL objective             | Downstream               | You write         |
|----------------------|---------------------------|--------------------------|-------------------|
| Neuroscience (EEG)   | VICReg (two-view)         | classification probe     | loader + augment  |
| Audio / speech       | VICReg (two-view)         | classification probe     | loader + augment  |
| Physion (video)      | predictive (action-cond.) | forecasting / probe      | loader            |
| Maze / Two-rooms     | predictive (action-cond.) | planning / probe         | loader            |
| Factors of variation | predictive (action-cond.) | planning vs perturbation | env adapter       |
| Intuitive physics    | predictive (video)        | VoE energy gap           | paired-clip probe |
| Time series (LTSF)   | predictive                | forecasting / regression | loader + baseline |

### Pick predictive when...

- there is temporal / sequence structure to predict
- an action or control channel is available
- → video, planning, time series

### Pick VICReg when...

- you have strong, meaning-preserving augmentations
- there is no natural “next step” to predict
- → EEG, audio, point clouds

**Bonus tracks:** hierarchical world models · scaling the recipe to new fields · handmade (non-synthetic) datasets.

## Takeaway

One energy-based recipe, three settings, single-GPU budget — a testbed for iterating on JEPA design before scaling up.

### Regularization theory

when does each term become necessary?

### Hierarchical world models

multi-timescale prediction

## Learned cost / value

beyond distance-to-goal planning

Thanks — questions welcome.

![QR code linking to the GitHub repository.](6b7b3f3d6f9341906163682cf12d1ea1_img.jpg)

A square QR code with a black and white pixelated pattern. In the center of the QR code is a circular logo featuring a white silhouette of a cat's head, which is the GitHub Octocat logo.

QR code linking to the GitHub repository.

**[github.com/facebookresearch/eb\\_jepa](https://github.com/facebookresearch/eb_jepa)**

Three examples · single-GPU · documented for teaching