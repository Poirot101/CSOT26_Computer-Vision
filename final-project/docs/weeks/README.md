# Per-week documentation

The course ran six weeks, each building one layer of the final pipeline. These
documents tie the syllabus to this codebase: what each week's material in the
original repository contains, which concept it introduced, where that concept
ends up in `src/cv_mot/`, and what was actually learned.

| Week | Topic | Data in the original repo | Where it lands in this project |
|---|---|---|---|
| [1](week1.md) | Neural networks, CNNs, PyTorch | Fashion-MNIST test set + labels | Conceptual only — no module |
| [2](week2.md) | ResNet-18, batch norm, augmentation | EuroSAT train (23,011 imgs) + test | Conceptual only — no module |
| [3](week3.md) | Transfer learning | readme only | The reason YOLO works off-the-shelf |
| [4](week4.md) | YOLO detection, Kalman, Hungarian, SORT | 4 MOT17 sequences, 2,925 frames | `detect.py`, `kalman.py`, `matching.py`, `sort.py` |
| [5](week5.md) | DeepSORT, ByteTrack, occlusion handling | readme only (in root README) | `bytetrack.py`, `track.py` |
| [6](final-project.md) | Dense-crowd MOT — the deliverable | reuses Week 4 sequences | the whole pipeline + `metrics/` |

## The thread running through all six weeks

Each week replaced a component of the previous week's model while keeping the
shape of the problem:

```
Week 1   flatten pixels        -> MLP           -> one class label
Week 2   keep spatial layout   -> ResNet-18     -> one class label
Week 3   reuse ImageNet weights-> frozen backbone + new head -> one class label
Week 4   swap the head again   -> detection head -> many boxes per image
Week 5   add time              -> boxes + motion -> persistent identities
Week 6   add density           -> same pipeline, 10x the people
```

The single most useful idea is the one Week 3 introduces and Week 4 reuses
without comment: a pretrained backbone plus a task-specific head. YOLOv8 is not
a different kind of model from the ResNet of Week 2 — it is the same transfer
learning move with a head that emits box coordinates instead of class scores.
