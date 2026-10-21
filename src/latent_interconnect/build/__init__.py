"""Construction stages upstream of training.

- ``capture``: receiver-state capture.
- ``ridge``: ridge initialization.
- ``logistic``: the L2 logistic fits of the supervised logistic baseline (``evaluation.logistic_baseline``).
- ``init_state``: the one initial-state procedure (the ``init`` stage; the trainer's step 0 calls it too).
- ``homology``: MMseqs2 homology searches, declared and recorded.
- ``train_selection``: the training objects, selected from the train split; its config schema is
  ``selection_spec`` and its evidence ``selection_evidence``.
- ``item_selection``: an item set drawn as whole clusters (the ``items`` stage).
- ``zero_shot``: the screened training set.
- ``molecule_detectors``: the molecule domain's exposure detectors for the screen (``chebi_role``, ``formula_mass``).

The layer-selection tools live in ``evaluation.layer_selection``, since they score.
"""

from latent_interconnect.build import molecule_detectors as _molecule_detectors  # registers two detector kinds

del _molecule_detectors
