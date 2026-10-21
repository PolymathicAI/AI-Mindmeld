"""Entry points, one per stage: ``python -m latent_interconnect.run.<stage> --config <yaml> [--pin N [--repin R]]``.

Each also takes ``[--destination N]``, ``[--option K=V]`` and ``[--purpose anchor]``; ``pin`` and ``read_scan_report``
(below) are commands, not stages.

The record kind of a stage run is the stage name. Outputs are named (artifact kinds in parentheses):

- ``pin`` (a command, not a stage; ``latent_interconnect.run.pin``): pin an existing record's outputs by record id,
  as ``--pin`` would have, to adopt a recorded execution without recomputing it.
- ``read_scan_report`` (a command, not a stage; ``latent_interconnect.run.read_scan_report``): print a read scan's
  layer combinations next to its single layers; writes nothing.
- ``domain``: config ``latent_interconnect.run.domain.DomainBuild``;
  outputs ``domain_data`` (domain_data).
- ``molecule_text``: config ``latent_interconnect.domains.molecule_text.MoleculeTextSpec``;
  outputs ``text`` (molecule_text).
- ``extract``: config ``latent_interconnect.run.extract.ExtractConfig``;
  outputs ``embeddings`` (embeddings).
- ``capture``: config ``latent_interconnect.build.capture.CaptureSpec``;
  outputs ``states`` (embeddings).
- ``ridge``: config ``latent_interconnect.build.ridge.RidgeSpec``;
  outputs ``ridge`` (ridge_init: a production ridge, or with ``exclude`` a study's held-out scan ridge).
- ``init``: config ``latent_interconnect.build.init_state.InitSpec``;
  outputs ``step_000000`` (checkpoint).
- ``homology``: config ``latent_interconnect.build.homology.HomologySearchSpec``;
  outputs ``hits`` (homology).
- ``train_selection``: config ``latent_interconnect.build.train_selection.TrainSelectionSpec``;
  outputs ``train_ids``, ``added`` and ``parents`` (item_set).
- ``training_set``: config ``latent_interconnect.build.zero_shot.TrainingSetSpec``;
  outputs ``training_set`` (training_set).
- ``items``: config ``latent_interconnect.build.item_selection.ItemSelectionSpec``;
  outputs ``item_set`` (item_set).
- ``xdom_items``: config ``latent_interconnect.build.cross_domain_items.CrossDomainItemSpec`` (a Rhea cross-domain
  design over its familiar/novel 2x2 cells: swap quads, ranking or direction); outputs ``items-<cell>`` per cell with an
  item (item_set), and the object lists ``proteins``, ``molecules`` and ``molecule_pool`` (item_set). With ``retained``
  the kept units of already built cells and the two lists of their objects: nothing is drawn.
- ``xdom_bdb_items``: config ``latent_interconnect.build.bindingdb_items.BindingDBItemSpec`` (the BindingDB
  familiar/novel 2x2); outputs ``molecule_domain`` (domain_data: the evaluation-only molecule data),
  ``items-<cell>`` per cell with a quad (item_set), and the object lists ``proteins`` and ``molecules`` (item_set).
  With ``retained`` the same outputs for the kept quads of already built cells: nothing is drawn.
- ``variant_items``: config ``latent_interconnect.build.variant_items.VariantItemSpec`` (the protein-mutation
  evaluation cohorts of one training contract, each on one axis: fixed discordant pairs or a sampled effect cohort,
  with each item's partners); outputs ``items-<cohort>`` per cohort (item_set) and the object list ``objects``
  (item_set) of every object their conditions write.
- ``scan``: config ``latent_interconnect.evaluation.layer_selection.ScanRunConfig`` (an injection scan or a layer-set
  comparison); outputs ``numbers`` (numbers: the summary with intervals) and ``scores`` (scores).
- ``read_scan``: config ``latent_interconnect.evaluation.read_scan.ReadLayerScanSpec`` (CPU; options ``shards``,
  ``shard``: a static split over workers); outputs ``numbers`` (numbers) and ``table`` (read_scan).
- ``logistic_baseline``: config ``latent_interconnect.evaluation.logistic_baseline.LogisticBaselineSpec`` (CPU; an
  encoder read's supervised logistic baseline on one Yes/No suite); outputs ``scores`` (scores: the suite's
  table shape, which ``grouped_contrasts`` reads as a suite record's) and ``numbers`` (numbers: per task and penalty).
- ``study``: config ``latent_interconnect.evaluation.layer_selection.LayerSelectionStudy``;
  outputs ``injection/numbers``, ``injection/scores`` (numbers, scores: the injection scan of the held-out scan
  ridge's predictions; a domain's own-state reference scan is a ``scan`` record).
- ``train``: config ``latent_interconnect.training.config.RunConfig``;
  outputs ``step_<6 digits>`` per committed step (checkpoint), ``log`` (train_log); one ``suite`` record per in-job
  evaluation, purpose ``monitor``.
- ``achieved``: config ``latent_interconnect.training.config.AchievedSpec`` (a stopped run's publication, on CPU);
  outputs ``step_<6 digits>`` per checkpoint the run's resume state holds as committed (checkpoint) and
  ``log`` (train_log, up to that step): a record of its own, never the run's, so a later completion pins as a fit.
- ``measure``: config ``latent_interconnect.training.measure.MeasureSpec``;
  outputs ``measurements`` (batch_measurement).
- ``suite``: config ``latent_interconnect.evaluation.suites.SuiteRunConfig``;
  outputs ``scores`` (scores), or ``generations`` (generations) for a generation suite, and ``bank/<argument>`` per
  write bank built.
- ``analysis``: config ``latent_interconnect.evaluation.analysis.AnalysisSpec``;
  outputs ``result`` (analysis).

Every entry point builds a :class:`~latent_interconnect.workspace.Workspace`, loads its config with the workspace's
pins, runs the stage, and writes the one record of the stage run
(:meth:`~latent_interconnect.workspace.Workspace.record`). The stage functions of the lower layers commit artifacts and
return them; they write no records. A stage returns ``latent_interconnect.run._main.StageResult``: its outputs, the
artifacts and records it read, and how its backend executed. The record's purpose follows its inputs
(:meth:`~latent_interconnect.workspace.Workspace.purpose`); ``--purpose anchor`` declares it ``anchor`` (unknown
lineage, ``run._main.stage_main``). The entry point prints the record's computation key and the output digests. With
``--pin NAME`` it pins ``NAME/<output>`` to each output, ``NAME`` too when there is exactly one output, and
``NAME@record`` to the record's computation key; each stage's naming of its outputs is its ``Stage.outputs``. A stage
config declares the name it is pinned under (``config.StageConfig.destination``, execution metadata outside its
identity), and a ``--pin`` other than that destination is refused before the stage runs (``run._main.check_pin``).
``--destination NAME`` overrides it for the run and pins under ``NAME`` as ``--pin NAME`` would, so ``--pin`` is then
not needed (a ``--pin`` other than ``NAME`` is refused); the record keeps the override in its provenance's
``execution`` (``destination.used``, ``destination.declared``). Which config makes a pin is
``latent_interconnect.config_index``. Pins are never overwritten: a name that already names another target is refused
(exit 3; the record and outputs stand) unless ``--repin REASON`` declares the re-pin, which
the pins history keeps (``Workspace.pin_all``); a rerun of a computation already recorded is pinned only with
``--repin`` (exit 2 otherwise), which declares which execution its names mean (``run._main.adopt``). Artifact pins and
record pins are typed, and a config field refuses a pin of the other kind (``config.ArtifactDigest``,
``config.RecordKey``). ``--option NAME=VALUE`` (repeatable) passes an execution option to the stage (a pass size, a
memory budget, a weights directory, an interpreter), never a config field
(``latent_interconnect.run._main.stage_options``); it is recorded in the provenance, not the identity. This package
holds the wiring; the stage functions live in the modules that own them.
"""
