"""
src — shared modules for multi-label CAM interference study.

Modules
-------
model           DenseNet169MultiLabel classifier
cam_engine      Gradient-based CAMEngine (GradCAM / LayerCAM / FPN variants)
scorecam_engine Gradient-free ScoreCAMEngine with baseline subtraction
evaluation      IoU, bootstrap CI, cardinality analysis, interference matrix
preprocessing   Image transforms, inference helpers, CAM overlay utilities
monusac_utils   MoNuSAC XML parser, data loading, patient-level split
"""
