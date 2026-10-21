# <Connector name and version>

Host this card with your connector. Link it from your LIP library submission.

- **Description:** What does the connector do, and which inputs does it accept?
- **Maintainer:** Repository and contact or issue tracker.
- **Models:** Specialist and receiving LLM names and revisions; tested AI Mindmeld version.
- **Downloads:** Versioned links to connector weights and configuration, with checksums.
- **Training:** Data sources, tasks, splits, and what was trained or frozen. Link details and
  distinguish connector training from model pretraining.
- **Injection layout:** Specialist read layers and pooling; object slots; receiving LLM write layers
  and numbering convention; `kind`, `kind_params`, token count and placement, per-token write layers, and update rule. Link the exact configuration.
- **Usage:** Installation and a working inference example, including the code for specialist
  loading and injection computation.
- **Licenses:** Code, connector weights, and upstream model/data requirements.
- **Results and limits:** Link evaluation or example outputs; state known limitations and untested uses.
