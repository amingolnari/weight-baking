"""Topology-aware weight baking: a local LLM analyst reads topological descriptors of one layer of a frozen target model, proposes a
geometric transformation, and the verified transformation is compiled into the target's weights by closed-form ridge regression,
layer by layer, without gradient descent.

Importing the package has no side effects (no folders, no printing, no model or GPU work); everything starts from ``topobake.cli.main``.
"""

__version__ = "0.1.0"
