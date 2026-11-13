# Planner Module

This module will contain the laundry grouping algorithms.

Planned methods:
1. color-only heuristic;
2. deterministic top-1 planner;
3. deterministic full-constraint planner;
4. uncertainty-aware planner;
5. oracle planner using ground-truth attributes.

All planners should consume structured `GarmentProfile` objects from `code/schema.py`.

The uncertainty-aware planner is the main proposed method. Its purpose is to make more conservative decisions when one or more garment attributes have low confidence.
