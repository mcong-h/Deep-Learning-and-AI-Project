# Code Structure

The codebase is organized around the research pipeline rather than the final user interface.

```text
code/
├── schema.py
├── perception/
├── planner/
├── evaluation/
└── app/
```

## Modules

### `schema.py`
Defines the shared structured representation of a garment, including predicted attributes, confidence scores, care constraints, and optional ground-truth information for evaluation.

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
