"""Proof of concept of the Adaptive Intelligence Engine described in the BRD.

A small agent plans a task, and for every step an adaptive router uses the fine-tuned Clef model to
decide what kind of work the step is, then a transparent policy picks the cheapest sufficient model.
"""

ROUTER_VERSION = "aie-poc-0.1.0"

