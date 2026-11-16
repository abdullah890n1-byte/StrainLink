"""Experimental AI failure-zone model: an optional add-on to StrainLink's analysis.

A machine-learning model (gradient-boosted trees, trained on SYNTHETIC scenes
measured with StrainLink's own DIC) gives the probability that each measured
point lies in a crack or shear band. It reads a finished analysis and writes
its own files beside it; it never changes StrainLink's results.
"""
