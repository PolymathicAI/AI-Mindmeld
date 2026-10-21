# Examples and guides

[Project overview](../../README.md) · [LIP library](../../library/README.md) · [Connectors on Hugging Face](https://huggingface.co/sgolkar/ai-mindmeld-connectors) · [Data on Hugging Face](https://huggingface.co/datasets/sgolkar/ai-mindmeld-data)

Start with the [Quickstart](../README.md), then choose a guide for your task.

## Use AI Mindmeld

- [Quickstart](../README.md): Install the package, download connectors, ask questions, evaluate, and run a small training example.
- [Interactive chat (web GUI)](../guides/chat.md): Set up browser chat and the HTTP API.
- [Chat CLI demo](../demos/README.md): Preview, edit, and send requests to a running chat server,
  with [input sources](../demos/README.md#input-sources).
- [LIP library](../../library/README.md): Browse connectors, scores, training descriptions, injection layouts, and versioned downloads.
- [Full data inventory](../../release/data_full.md): Choose reusable training, evaluation, and inference inputs; download and verify them, and check compatibility and upstream terms.
- [Example training data](../data/README.md): Read the source and license of the 32-protein training example.
- [Artifact licenses](../../LICENSE-ARTIFACTS.md): Check the grant for connector weights and selected data, attribution, and upstream restrictions.

## Work with the code

- [Extending the package](../guides/extending.md): Add specialist encoders, LLMs, and connector kinds, with a worked training example.
- [Choosing layers](../guides/choosing_layers.md): Read layer-selection scans and choose specialist read layers and receiving LLM write layers.
- [Batching and determinism](../guides/batching.md): Understand how batching and numerical precision affect scores and reproducibility.
- [Packaging and sharing a connector](../guides/contributing-a-connector.md): Package a checkpoint, check its installation, host it, and submit a library listing.
- [Contributing](../../CONTRIBUTING.md): Submit a connector listing or a focused pull request to improve the repository.
- [Connector card template](../../library/CONNECTOR_CARD_TEMPLATE.md): Describe your connector's models, training data, injection layout, downloads, and limitations.
- [Developer checks](../../CONTRIBUTING.md#developer-checks): Run lint, formatting, and pytest checks, with optional model-specific gates.
