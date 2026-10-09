# Week 3 — Transfer learning, feature extraction and fine-tuning

## What the original repo contains

`Week 3/` is **readme only** (12 KB) — no data of its own. The week reuses the
EuroSAT dataset from Week 2 and replaces the from-scratch ResNet with a
pretrained one.

That absence is itself the point: the week adds no data because the whole lesson
is that you need *less* data when you start from pretrained weights.

## The concepts

**Why pretrained features transfer.** A network trained on ImageNet learns a
hierarchy: early layers detect edges, corners and colour gradients; middle layers
detect textures and parts; late layers detect object-specific configurations. The
early and middle layers are not about ImageNet's 1,000 classes — they are about
how natural images are structured. Satellite imagery shares that structure even
though it shares no classes.

**Feature extraction versus fine-tuning** — the decision rests on two axes:

| | small dataset | large dataset |
|---|---|---|
| **similar domain** | freeze backbone, train head | fine-tune everything |
| **different domain** | freeze early layers, train later ones + head | fine-tune everything |

Freezing means the backbone's weights get no gradient; only the replaced
classification head trains. This is fast, needs little data, and cannot destroy
the pretrained features.

**Learning rate is the dangerous parameter.** Pretrained weights sit in a good
minimum. A learning rate appropriate for random initialisation will take a large
first step and wreck them — the model gets *worse* than the frozen version. Hence
discriminative learning rates: a small rate (or zero) for the backbone, a larger
one for the freshly-initialised head, which has no information to protect.

## Where this lands in the final project

This is the week the final project leans on hardest, even though it contributes
no code.

**YOLOv8 is transfer learning.** `cv_mot/detect.py` loads `yolov8s.pt` — weights
pretrained on COCO — and runs it on MOT17 pedestrian footage it has never seen,
with **no training at all**. It works for exactly the reason Week 3 explains:
COCO contains a `person` class, and the features that locate a person in a COCO
photo locate a person in a surveillance frame.

That is why the detector is restricted to a single class:

```python
COCO_PERSON_CLASS = 0
...
classes=[COCO_PERSON_CLASS]
```

The other 79 COCO classes are not just unnecessary, they are harmful — a tracked
"handbag" or "car" is a false positive against pedestrian ground truth.

**The head-swap framing.** Week 2 built ResNet-18 to emit 10 class scores. Week 3
replaced that head to emit EuroSAT's 10 classes. Week 4's YOLO replaces it again
with a head emitting box coordinates *and* class scores for every spatial
position. Same backbone idea, different output structure — which is why the Week
4 readme can call YOLO "the same idea taken further" and be exactly right.

## What was learned

The strongest result of the whole project — a detector achieving usable recall on
MOT17 with zero task-specific training — is a transfer learning result, not a
tracking one. Knowing *when not to train* was worth more here than any
architectural knowledge: the available compute (4 CPU cores, no GPU) made
fine-tuning a detector impossible, and the project still works because pretrained
features were sufficient.

It also sets up the honest limit stated in [LEARNINGS.md](../LEARNINGS.md): a
COCO-pretrained detector is *generic*. A detector fine-tuned on crowded
pedestrian data would do better, and that remains the largest untapped gain.
