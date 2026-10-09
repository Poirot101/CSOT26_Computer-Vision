# Week 2 — Deep networks, ResNet-18 and data augmentation

## What the original repo contains

`Week 2/` (101 MB) — the largest week by data volume:

| File | Contents |
|---|---|
| `readme.md` | Action plan |
| `labels.txt` | EuroSAT class map: `AnnualCrop:0, Forest:1, HerbaceousVegetation:2, Highway:3, Industrial:4, Pasture:5, PermanentCrop:6, Residential:7, River:8, SeaLake:9` |
| `train.zip` | 86 MB — 23,011 training images in 10 class folders |
| `test_set.zip` | 15 MB |
| `ground_truth.csv`, `sample_submission.csv` | 64 KB each |

Class counts in `train.zip` are uneven (2,600 for Forest/AnnualCrop/SeaLake/
HerbaceousVegetation/Residential down to 1,600 for Pasture), which matters: plain
accuracy flatters a model on an imbalanced set, and per-class accuracy or a
confusion matrix is the honest reading. The week's readme asks for exactly that.

The dataset jumps from 28×28 grayscale clothing to 64×64 RGB satellite imagery.

## The concepts

**Two distinct failures of depth.** Worth keeping separate because they have
different fixes:

- *Vanishing gradients* — gradient magnitude shrinks multiplicatively through
  layers, so early layers barely update. Normalisation and better activations
  help.
- *Degradation* — a deeper network does worse on **training** data than a
  shallower one. This is not overfitting and not a gradient-magnitude problem.
  It means the optimiser cannot find a solution that the deeper architecture
  provably contains (it could always set the extra layers to identity).

**Skip connections** address degradation directly. A residual block computes
`F(x) + x`, so the layer learns a *correction* to its input rather than a whole
transformation. Identity is then the easy default — reachable by driving `F`
toward zero — instead of something the optimiser must construct.

**Batch normalisation** standardises activations per mini-batch, which smooths
the loss surface and permits higher learning rates. It also makes the network's
behaviour depend on batch composition during training, which is why it switches
to running statistics at inference.

**Augmentation as regularisation.** Flips, rotations, crops and colour jitter
state invariances that should not change the label. For satellite imagery a
vertical flip is fine (there is no canonical "up" from orbit); for Fashion-MNIST
it is not (an upside-down boot is not a boot). The transform list encodes a claim
about the domain, so copying one between datasets without thinking is a bug.

## Where this lands in the final project

Still no classifier module — but one idea transfers directly and is visible in
the code:

**Normalisation choices encode priors about scale.** `cv_mot/kalman.py` scales
every noise standard deviation by box height:

```python
self._std_weight_position * h      # position noise grows with object size
```

A pedestrian 400 px tall crosses many more pixels per frame than one 40 px tall.
A single fixed covariance is wrong for both. Making the noise proportional to `h`
is the same instinct as normalising activations — put the quantity on a scale
where one parameter setting is valid across the whole range. It is why one
configuration works for near and far pedestrians in the same frame.

The aspect-ratio component is deliberately *excluded* from that scaling
(`aspect_std` is a small constant) because a walking person's width-to-height
ratio is roughly fixed while their pixel size is not.

## What was learned

Distinguishing *optimisation* failures from *capacity* failures changes what you
try next: more data and augmentation fix overfitting, while skip connections fix
degradation, and applying one remedy to the other problem wastes effort. The same
diagnostic split reappears in the final project as `DetA` versus `AssA` — knowing
whether the pipeline is losing points to detection or to association is what
makes tuning directed rather than random.
