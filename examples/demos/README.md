# Chat CLI demo

Preview or send protein, molecule, and combined-object questions with the Python 3.11+ client; no extra
packages are needed. Live requests need an existing AI Mindmeld chat
server with the matching LLM, connectors, and specialist encoders; set one up with
[Interactive chat (web GUI)](../guides/chat.md). For local CPU inference, use the [Quickstart](../README.md).

Run these commands from the repository root. They use `http://127.0.0.1:8811` by default; add
`--url ADDRESS` for another server and `--password-file /path/to/password` for authentication.
Use HTTPS or a local tunnel for a remote server.

## Preview and check availability

```bash
python examples/demos/chat.py examples/demos/requests/protein.json --show-request
python examples/demos/chat.py --list
```

The preview is offline. `--list` contacts the server and shows its deployed models. These requests need
`esmc-600m` and/or `chemfm-1b`. In the JSON, `messages[0].content` is the question and `placeholders`
binds each object slot to its connector and raw sequence or SMILES. Raw inputs require the server's
matching specialist encoder environment and model weights. To ask molecule questions without running
ChemFM locally, use the stored-vector examples in [Quickstart step 4](../README.md#4-ask-a-question).

## Ask a question

```bash
python examples/demos/chat.py examples/demos/requests/protein.json
python examples/demos/chat.py examples/demos/requests/molecule.json
python examples/demos/chat.py examples/demos/requests/composition.json
```

The inputs are ubiquitin, aspirin, and aspirin paired with human COX-1. The composition request asks:

> Could the molecule [[M1]] inhibit the protein [[P1]]? Explain what mechanism would account for inhibition,
> using the protein's likely function and the molecule's chemical groups.

Edit the questions and `placeholders.P1.value` / `placeholders.M1.value` to try your own inputs.
[Input sources](#input-sources) include COX-1's canonical 599-residue sequence, UniProt P23219.

## Save a transcript

```bash
python examples/demos/chat.py examples/demos/requests/protein.json --output demo-output/protein.json
```

The transcript stores the request, server configuration, events, and completion status, including partial
output after a failure. Passwords are excluded; existing files are not overwritten. Answer text streams
to stdout, status to stderr, and reasoning events are stored separately.

The requests use greedy generation and a 256-token cap. `finish_reason: length` means the cap was reached. `--timeout` sets the socket timeout (default 900 seconds).

## Input sources

The standalone protein example uses the ubiquitin sequence. The molecule example uses aspirin, represented
as `CC(=O)OC1=CC=CC=C1C(=O)O`.

The composition example pairs aspirin with human cyclooxygenase-1 (COX-1 / PTGS1). Its protein input is the
complete 599-amino-acid canonical sequence from [UniProtKB P23219](https://www.uniprot.org/uniprotkb/P23219/entry),
sequence version 2, retrieved on October 19, 2026 from the
[UniProt FASTA endpoint](https://rest.uniprot.org/uniprotkb/P23219.fasta).
UniProt documents aspirin inhibition in this entry. The question asks the LLM to infer the relationship and
mechanism from the two representations; it supplies neither object's name nor the known answer.

UniProt's [attribution and license information](https://www.uniprot.org/help/license)
applies to its sequence data.
