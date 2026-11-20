# Code Structure

The codebase is organized around the research pipeline rather than the final user interface.

```text
code/
├── schema.py
├── dataset.py
├── encoder.py
├── compatibility_model.py
├── train.py
├── perception/
├── planner/
├── evaluation/
└── app/
```

## First Training Prototype

The first runnable prototype trains a small neural network to predict whether two garments are unsafe to wash together.

For now, it uses:
- synthetic garment records;
- rule-generated safe/unsafe pair labels;
- a simple structured garment encoder;
- a trainable pairwise MLP classifier.

This is intentionally a prototype. It lets us validate the training pipeline before connecting real garment images, care-label recognition, calibration, and the final laundry planner.

### Run

From the repository root:

```bash
cd code
python train.py
```

The script will:
1. create toy garments;
2. split garments into train/validation/test sets;
3. generate garment pairs and safe/unsafe labels;
4. encode each garment as a feature vector;
5. train the pairwise compatibility model;
6. print validation and test performance;
7. save the trained checkpoint to `results/compatibility_model.pt`.

PyTorch is required. The optional CLIP image encoder in `encoder.py` additionally requires `transformers` and `pillow`, but CLIP is not used by the default toy run.

## Main Files

### `schema.py`
Defines the shared structured representation of a garment, including predicted attributes, confidence scores, care constraints, and optional ground-truth information for evaluation.

### `dataset.py`
Creates garment records, splits data by garment to reduce leakage, forms garment pairs, and produces temporary weak/synthetic safe/unsafe labels.

### `encoder.py`
Contains:
- a structured encoder used by the first runnable prototype;
- an optional frozen pretrained CLIP image encoder for later real-image experiments.

### `compatibility_model.py`
Defines the trainable `CompatibilityMLP`. It receives representations of two garments and outputs one logit representing their predicted incompatibility.

### `train.py`
Runs the full first-stage training loop using binary cross-entropy loss and Adam optimization.

### `perception/`
Will contain garment-image and care-label recognition components. Its job is to convert raw images into a structured `GarmentProfile`.

Expected outputs include:
- garment type;
- color;
- material;
- wash/care constraints;
- confidence values or probability distributions.

### `planner/`
Will contain the decision-making methods:
- deterministic top-1 planner;
- deterministic full-constraint planner;
- uncertainty-aware planner;
- oracle planner.

The planners should consume structured garment profiles rather than raw images.

### `evaluation/`
Will contain experiment scripts and metrics, including:
- safety violation rate;
- unsafe-pair rate;
- number of loads;
- load utilization;
- regret relative to the oracle.

### `app/`
Will contain the interactive demo after the research pipeline is working. The app is intentionally separated from the core research logic.

## Design Principle

The project separates **perception** from **planning** so that we can explicitly study how recognition uncertainty affects downstream laundry decisions.
