# Perception Module

This module will convert garment and care-label images into structured garment profiles.

Planned responsibilities:
- garment photo → color and garment-type predictions;
- care-label photo → material composition and wash/care constraints;
- expose confidence values or probability distributions;
- return a shared `GarmentProfile` defined in `code/schema.py`.

The first implementation should favor information explicitly available on the care label over guessing exact material composition from garment appearance alone.
