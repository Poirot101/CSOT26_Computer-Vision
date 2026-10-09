# Week 1 — Neural networks, CNNs and PyTorch foundations

## What the original repo contains

`Week 1/` (128 KB):

| File | Contents |
|---|---|
| `readme.md` | The week's action plan |
| `labels.txt` | 10 Fashion-MNIST class names, `0 = T-shirt/top` … `9 = Ankle boot` |
| `test_set.zip` | 101 KB of 28×28 grayscale test images |
| `sample_submission.csv` | `image_id,label` — 50 rows, all predicting class 0 |
| `ground_truth.csv` | The same 50 ids with true labels |

The task: classify clothing images, submit `image_id,label`, score on a
leaderboard.

## The concepts

**Perceptron → MLP.** A linear layer computes `y = Wx + b`. Stacking them with
nothing in between is pointless: a composition of linear maps is linear. The
non-linearity (ReLU) is what makes depth buy anything.

**Backpropagation** is the chain rule applied to a computation graph, run
backwards so each intermediate derivative is computed once rather than
recomputed per parameter. PyTorch's `autograd` builds that graph as the forward
pass executes.

**Why flattening is wrong.** An MLP on Fashion-MNIST flattens 28×28 into 784
values, which destroys the fact that pixel 0 is adjacent to pixel 1 but not to
pixel 100. The network has to relearn adjacency from data. A convolution has
that adjacency built into its structure: one small kernel slides over the image,
so a feature detected in the top-left uses the same weights as one in the
bottom-right. That weight sharing is both a massive parameter reduction and a
statement about the problem — *translation does not change what something is*.

**Stride, padding, pooling.** Stride sets how far the kernel moves; padding
preserves spatial size at the border; pooling discards spatial resolution in
exchange for a degree of translation invariance and a cheaper next layer.

## Where this lands in the final project

No module in `src/cv_mot/` implements a classifier — the final project consumes
a pretrained detector rather than training one. The transferable content is:

- **`.to(device)` discipline.** Every tensor in an operation must live on the
  same device. `cv_mot.detect.YoloDetector` takes a `device` argument and
  passes it through to Ultralytics for exactly this reason, and
  `boxes.cpu().numpy()` in `detect.py` is the same lesson in reverse: results
  must come *back* before NumPy touches them.
- **Dataset/DataLoader thinking** — a lazily-iterated sequence of samples with
  batching. `cv_mot.detect.Detector` is that interface, narrowed to one job:
  `run(seq)` yields one frame's `Detections` at a time, so a 1,050-frame
  sequence never needs to fit in memory.

## What was learned

Convolution is not a trick for efficiency; it is an encoding of a prior about
images. Recognising which assumptions a model architecture *builds in* versus
which it must *learn from data* turned out to be the reusable skill — it is the
same reasoning that later justifies a Kalman filter (constant velocity is a
built-in prior about how people move) over learning motion from scratch.
