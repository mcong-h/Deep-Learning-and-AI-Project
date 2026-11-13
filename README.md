# Uncertainty-Aware AI Laundry Load Planning

## Project Overview
This project investigates how uncertainty in AI-based garment and care-label recognition affects downstream laundry planning.

Existing work has studied garment/care-label recognition and laundry optimization, but these stages are often treated separately. In practice, recognition models can make uncertain or incorrect predictions. A laundry planner that treats every top-1 prediction as fully reliable may therefore create unsafe garment groupings.

Our goal is to build and evaluate an **uncertainty-aware laundry planner** that uses prediction confidence when deciding how garments should be grouped into washing loads.

## Core Research Question
**Can uncertainty-aware planning reduce unsafe laundry decisions caused by imperfect garment recognition without substantially increasing the number of washing loads?**

## Hypothesis
A planner that incorporates prediction confidence will reduce unsafe garment groupings compared with a deterministic planner that treats every top-1 prediction as certain, while increasing the number of loads only modestly.

## Motivation
A practical laundry assistant must do more than recognize garments. It must make decisions from imperfect predictions.

For example, if a model predicts that a sweater is:
- cotton: 54%
- wool: 41%
- other: 5%

a deterministic system may simply treat the sweater as cotton. An uncertainty-aware system can recognize that the prediction is ambiguous and choose a more conservative grouping or washing setting.

This creates a **safety-efficiency tradeoff**:
- safer decisions may require more conservative grouping;
- fewer loads are more efficient but may increase risk when recognition is uncertain.

## Proposed System
1. **Garment image** — predict color and garment type.
2. **Care-label image** — extract material composition and washing constraints from text/symbols.
3. **Structured garment profile** — store predicted attributes together with confidence scores.
4. **Laundry planner** — compare deterministic planning with uncertainty-aware planning.
5. **Evaluation** — measure safety and efficiency against an oracle planner that uses ground-truth attributes.

## Planned Baselines
- **Color-only heuristic**: groups garments using simple color rules.
- **Deterministic planner**: uses predicted garment attributes but treats top-1 predictions as certain.
- **Deterministic full-constraint planner**: uses predicted color, material, and care-label constraints without confidence information.
- **Oracle planner**: uses ground-truth garment attributes and serves as an upper-bound reference.
- **Our method — uncertainty-aware planner**: incorporates prediction confidence into grouping decisions.

## Planned Evaluation Metrics
- Safety violation rate
- Unsafe-pair rate
- Number of washing loads
- Average load utilization
- Decision regret relative to the oracle planner

## Related Work
Our project builds on three relevant research directions:
1. garment and care-label recognition;
2. AI-assisted laundry sorting and laundry optimization;
3. uncertainty-aware / selective prediction.

See [docs/related_work.md](docs/related_work.md) for the current literature summary.

## Repository Structure
```
.
├── code/
├── data/
├── docs/
│   ├── research_plan.md
│   └── related_work.md
├── poster/
├── report/
├── results/
├── LICENSE
└── README.md
```

## Project Status
Research framing and literature review in progress. Implementation has not yet begun.

## Course
Applied Deep Learning and AI — Fall 2026
