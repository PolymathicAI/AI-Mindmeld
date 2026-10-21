"""Download released connector and data bundles into a data root the stages, ``examples.ask`` and chat read.

    python -m latent_interconnect.release_download \
        --connector gemma-3-4b-it__esmc-600m --data protein-domain --dest <dir>

The release's bundles are public, versioned downloads from two Hugging Face repositories: the model repository
:data:`CONNECTORS_REPO` and the dataset repository :data:`DATA_REPO` (every data bundle, the two minimal eval sets
among them). Downloading them needs no login; Hugging Face authentication is needed only for gated upstream models
the receivers load, such as Gemma, and for a private community repository one is authorized to read. The default
revisions
(:data:`CONNECTORS_REVISION`, :data:`DATA_REVISION`) are the release commits; ``--revision`` and ``--data-revision``
override them. Each holds an
index at its root (``connectors.json``, ``data.json``) and one directory per bundle (``<name>/``) with its
``bundle.json``: the bundle's artifacts (each one's id, kind, directory in the bundle and per-file sha256 and size)
and the pins that name them. A download checks the chain: the index's sha256 of ``bundle.json``, then each file's
sha256 and size, then the artifact's manifest digest, which is its id. It commits each artifact to
``<dest>/lip/artifacts`` through :meth:`ArtifactStore.commit` (the identity it computes must be the bundle's), pins
the bundle's names in ``<dest>/lip/pins.yaml`` (``Workspace.pin_all``) and keeps each ``bundle.json`` as
``<dest>/lip/bundles/<name>.json``, so ``LIP_DATA=<dest>`` serves the suite configs that cite those pins. A file
that fails a check stops the download before
anything of its artifact is committed or pinned.

For each connector it installs, the command prints its map kind (every ``bundle.json`` states it as ``write.kind``),
its ``kind_params``, the input it reads (pooled vector or per-residue states), its tokens and write layers (taken
from the bundle's optional ``connector_kind`` field when it has one, as bundles made by
:mod:`latent_interconnect.bundle_connector` do, and otherwise derived from the installed ``connector.json``, as for
the released and DEMO bundles), the checkpoint's installed directory (the one ``examples/ask.py`` takes), the
``plugins`` it needs importable (the bundle's ``plugins``, else those its connector and encoder recipe name), and
where its receiver config is: a bundle that carries one
(``receiver_config``, :mod:`latent_interconnect.bundle_connector`) has it installed as
``<dest>/lip/receivers/<receiver>.yaml``, the file ``examples/ask.py`` takes; the released bundles' receivers are in
the checkout's ``configs/receivers/``.

A community connector installs the same way from its author's own model repository: ``--repo <user/repo> --revision
<commit> --connector <name>`` (a repository laid out as above, e.g. by :mod:`latent_interconnect.bundle_connector`).
A repository other than the release's needs ``--revision`` (and ``--data-revision`` for another data repository),
the full 40-hex commit its listing names, so what is installed is fixed; a branch or tag such as ``main`` is refused.

``--all-data`` selects every data bundle of the data index (the full-data bundles and the two minimal eval sets) except
DEMO bundles (group ``demo`` or a name ending in ``-demo``); it adds to any ``--data`` and ``--connector`` named.
``--dry-run`` prints the selected bundles with their sizes and the total, and exits without installing anything.

``--list`` prints every connector and data bundle the two indexes name, with its size (the index's ``bytes``; in GB,
or in MB below 0.1 GB) and the index's ``description`` when it has one, and exits without installing anything.
``--list --kinds`` also reads each connector's ``bundle.json`` (a small file, checked against the index) and prints
its kind, input (pooled or residues), token count and ``DEMO`` for a demo connector.

On success the command prints the pins it wrote, each connector's report (above) and one line
``verified N installed files (sha256)`` (every file of the installed artifacts, fetched or already present, checked
against its sha256). A failed check prints one
line naming the file and the Hugging Face cache entry to delete before a rerun, and exits non-zero.

Files come from the Hugging Face Hub (``huggingface_hub``, the ``hf`` extra). ``--local-source DIR`` reads
``DIR/connectors`` and ``DIR/data`` instead of the hub (for checking a local bundle).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from latent_interconnect.artifacts.records import RecordStore
from latent_interconnect.artifacts.store import MANIFEST_NAME, ArtifactStore, IntegrityError, Manifest, sha256_file
from latent_interconnect.config import Pin, dump_config
from latent_interconnect.paths import Paths, code_checkout
from latent_interconnect.workspace import Workspace

__all__ = [
    "BUNDLE_FILE",
    "CONNECTORS_REPO",
    "CONNECTORS_REVISION",
    "DATA_REPO",
    "DATA_REVISION",
    "BundleIntegrityError",
    "Fetch",
    "all_data_names",
    "connector_kind_text",
    "connector_summary",
    "download",
    "hub_fetch",
    "list_bundles",
    "local_fetch",
    "main",
]

BUNDLE_FILE = "bundle.json"
_COMMIT = re.compile(r"[0-9a-f]{40}")
CONNECTORS_REPO = "sgolkar/ai-mindmeld-connectors"
"""The model repository of the connector bundles."""
CONNECTORS_REVISION = "72a79954ec2780f49f96ccf3a4bdaf0d19642f6b"
"""The release commit of :data:`CONNECTORS_REPO`: 27 entries, the 25 step-2000 release connector bundles (the 19 paper
connectors, the two ESM-C 6B affine connectors of Kimi K2.6 and Gemma 3 12B, Kimi K2.6's ESM-C 6B mlp connector, the
two ChemFM-3B affine connectors of Gemma 3 4B and 12B and Kimi K2.6's ChemFM-3B mlp connector) and two DEMO
connectors (``*-demo``, software demonstrations documented in ``examples/README.md``, not results)."""
DATA_REPO = "sgolkar/ai-mindmeld-data"
"""The dataset repository of the data bundles (domains, training sets, item sets, embeddings, ridge starts)."""
DATA_REVISION = "a27d7808625794b31e44ad9b909816c17748b8f0"
"""The release commit of :data:`DATA_REPO`: 41 entries in the index ``data.json``: the 38 bundles of the full release
(the two domains, two training sets and two item-set bundles, the eight encoders' embeddings and the 24 ridge starts the
release connectors' run and suite configs read), the two minimal eval sets ``protein-esmc-600m-mini`` and
``molecule-chemfm-1b-mini``, and one DEMO bundle (``*-demo``, a software demonstration documented in
``examples/README.md``, not results)."""
_MB_BELOW = 10**8
"""Sizes below 0.1 GB (10^8 bytes) are printed in MB."""
INDEXES = {"connector": ("connectors.json", "connectors"), "data": ("data.json", "data")}


class BundleIntegrityError(IntegrityError):
    """A bundle failed a check; ``kind`` and ``name`` say which, ``installed`` whether the copy in the data root did."""

    def __init__(self, message: str, kind: str, name: str, *, installed: bool = False) -> None:
        super().__init__(message)
        self.kind, self.name, self.installed = kind, name, installed


Fetch = Callable[[str], Path]
"""A file fetcher: path in the release repository → a local file holding it (read only)."""


def hub_fetch(repo: str, revision: str, repo_type: str = "model") -> Fetch:
    """Return a fetcher of one revision of a Hugging Face Hub repository (files land in the hub cache)."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415  (the optional `hf` extra)

    def fetch(path: str) -> Path:
        return Path(hf_hub_download(repo, path, revision=revision, repo_type=repo_type))

    return fetch


def local_fetch(root: Path) -> Fetch:
    """Return a fetcher of a local directory laid out as the release repository (tests only)."""

    def fetch(path: str) -> Path:
        p = root / path
        if not p.is_file():
            raise FileNotFoundError(f"{path} is not in {root}")
        return p

    return fetch


def _bundle(fetch: Fetch, kind: str, name: str) -> tuple[str, dict[str, Any]]:
    """Return a bundle's directory and its ``bundle.json``, checked against the index."""
    index_file, key = INDEXES[kind]
    index = json.loads(fetch(index_file).read_text())
    entries = {e["name"]: e for e in index[key]}
    if name not in entries:
        raise KeyError(f"no {kind} bundle {name!r} in {index_file}; it lists {sorted(entries)}")
    entry = entries[name]
    local = fetch(f"{entry['path']}/{BUNDLE_FILE}")
    if sha256_file(local) != entry["bundle_sha256"]:
        raise IntegrityError(f"{entry['path']}/{BUNDLE_FILE} differs from the sha256 {index_file} records")
    return str(entry["path"]), json.loads(local.read_text())


def _install_artifact(fetch: Fetch, base: str, art: dict[str, Any], store: ArtifactStore) -> int:
    """Fetch, check and commit one artifact (a no-op when the store already holds it intact); return its file count.

    Raises:
        IntegrityError: for a fetched file, manifest or commit that fails its check.
        BundleIntegrityError: (``installed``) for an already installed copy that fails :meth:`ArtifactStore.verify`.
    """
    sha = art["sha256"]
    if store.exists(sha):
        try:
            store.verify(store.ref(sha))
        except IntegrityError as err:
            raise BundleIntegrityError(str(err), "", "", installed=True) from err
        return len(store.manifest(store.ref(sha)).files)
    prefix = base if art["path"] in ("", ".") else f"{base}/{art['path']}"
    manifest = Manifest.load(fetch(f"{prefix}/{MANIFEST_NAME}"))
    if manifest.sha256() != sha:
        raise IntegrityError(f"{prefix}/{MANIFEST_NAME} does not hash to the artifact id {sha[:12]}")
    listed = {rel: (f["sha256"], int(f["bytes"])) for rel, f in art["files"].items()}
    if listed != dict(manifest.files):
        raise IntegrityError(f"{prefix}: {BUNDLE_FILE} and {MANIFEST_NAME} list different files")
    staged = store.staging(f"release-download-{sha}")
    for rel, (want, size) in sorted(listed.items()):
        src = fetch(f"{prefix}/{rel}")
        target = staged / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        got = sha256_file(target)
        if got != want or target.stat().st_size != size:
            raise IntegrityError(f"{prefix}/{rel}: sha256 {got[:12]}, the bundle records {want[:12]}")
    ref = store.commit(staged, manifest.kind, manifest.meta)
    if ref.sha256 != sha:
        raise IntegrityError(f"{prefix}: committed as {ref.short}, the bundle names {sha[:12]}")
    return len(listed)


def connector_summary(bundle: dict[str, Any], store: ArtifactStore, dest: Path) -> list[str]:
    """Return the install report of a connector bundle: kind, input, layout, plugins, receiver config.

    Installs the bundle's ``receiver_config``, when it has one, as ``<dest>/lip/receivers/<receiver>.yaml``. Its
    ``connector_kind`` and ``plugins``, when absent, are derived from the installed checkpoint's ``connector.json``.
    """
    from latent_interconnect.bundle_connector import connector_layout  # noqa: PLC0415  (pulls in the training code)
    from latent_interconnect.receivers.base import ReceiverSpec  # noqa: PLC0415
    from latent_interconnect.writes.checkpoint import CHECKPOINT_KIND  # noqa: PLC0415

    name = bundle["name"]
    license_lines = [
        f"  license (this bundle): {bundle.get('license', 'not stated')}",
        "  The project code is Apache-2.0; check this bundle's license and its upstream terms before use.",
    ]
    checkpoints = [
        store.path(store.ref(a["sha256"])).resolve() for a in bundle["artifacts"] if a.get("kind") == CHECKPOINT_KIND
    ]
    checkpoint_lines = [f"  checkpoint directory: {c}" for c in checkpoints]
    layout, plugins = bundle.get("connector_kind"), bundle.get("plugins")
    if layout is None or plugins is None:
        stored = [store.path(store.ref(a["sha256"])) / "connector.json" for a in bundle["artifacts"]]
        header = next((json.loads(p.read_text()) for p in stored if p.is_file()), None)
        if header is None:
            return [
                f"connector {name}: the bundle states no kind and holds no connector.json",
                *checkpoint_lines,
                *license_lines,
            ]
        layout = layout or connector_layout(header["spec"], int(header["depth"]))
        if plugins is None:
            plugins = sorted({*header["spec"].get("plugins", ()), *header["encoder"].get("plugins", ())})
    lines = [
        f"connector {name}: kind {layout['kind']} {json.dumps(layout['kind_params'], sort_keys=True)}, "
        f"reads {layout['input']}, {layout['tokens']} token(s), write layers {layout['write_layers']}",
        *checkpoint_lines,
        f"  plugins to have importable: {', '.join(plugins) if plugins else 'none'}",
        *license_lines,
    ]
    receiver = bundle["receiver"]["name"]
    if bundle.get("receiver_config") is not None:
        spec = ReceiverSpec.model_validate(bundle["receiver_config"])
        path = dest / "lip" / "receivers" / f"{receiver}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        dump_config(spec, path)
        lines.append(f"  receiver config: {path}")
    else:
        lines.append(f"  receiver config: not in the bundle; the release's is configs/receivers/{receiver}.yaml")
    return lines


def download(
    fetch: Fetch,
    dest: Path,
    connectors: Sequence[str] = (),
    data: Sequence[str] = (),
    data_fetch: Fetch | None = None,
    *,
    report: list[str] | None = None,
) -> dict[str, Pin]:
    """Install the named bundles under the data root ``dest`` and return the pins written.

    ``fetch`` reads the connectors' repository, ``data_fetch`` the data's (``fetch`` too when None). ``report``, when
    given, receives each connector's install report (:func:`connector_summary`).

    Raises:
        KeyError: for a bundle no index lists.
        BundleIntegrityError: for a file, manifest or bundle that fails its check.
    """
    return _download(fetch, dest, connectors, data, data_fetch, report=report)[0]


def _download(
    fetch: Fetch,
    dest: Path,
    connectors: Sequence[str],
    data: Sequence[str],
    data_fetch: Fetch | None,
    *,
    report: list[str] | None = None,
) -> tuple[dict[str, Pin], int]:
    """:func:`download`, also returning the number of artifact files checked against their sha256."""
    paths = Paths.from_env({"LIP_DATA": str(dest)})
    ws = Workspace(
        paths, ArtifactStore(paths.artifacts), RecordStore(paths.records), dest / "lip" / "pins.yaml", code_checkout()
    )
    pins: dict[str, Pin] = {}
    verified = 0
    for kind, names, source in (("connector", connectors, fetch), ("data", data, data_fetch or fetch)):
        for name in names:
            try:
                base, bundle = _bundle(source, kind, name)
                for art in bundle["artifacts"]:
                    verified += _install_artifact(source, base, art, ws.artifacts)
            except IntegrityError as err:
                installed = isinstance(err, BundleIntegrityError) and err.installed
                raise BundleIntegrityError(str(err), kind, name, installed=installed) from err
            kept = dest / "lip" / "bundles" / f"{name}.json"  # the recipe and provenance, for the user's own encoder
            kept.parent.mkdir(parents=True, exist_ok=True)
            kept.write_text(json.dumps(bundle, indent=1, sort_keys=True) + "\n")
            pins |= {k: Pin.model_validate(v) for k, v in bundle["pins"].items()}
            if kind == "connector" and report is not None:
                report.extend(connector_summary(bundle, ws.artifacts, dest))
    ws.pin_all(pins)
    return pins, verified


def _is_demo(entry: dict[str, Any]) -> bool:
    return entry.get("group") == "demo" or str(entry["name"]).endswith("-demo")


def all_data_names(data_fetch: Fetch) -> list[str]:
    """Every data bundle of the data index except DEMO bundles (``--all-data``), in index order."""
    index_file, key = INDEXES["data"]
    return [e["name"] for e in json.loads(data_fetch(index_file).read_text())[key] if not _is_demo(e)]


def _size(entry: dict[str, Any]) -> str:
    """An index entry's size: GB (10^9 bytes), or MB (10^6) below 0.1 GB."""
    if "bytes" not in entry:
        return "size not in the index"
    n = int(entry["bytes"])
    return f"{n / 1e9:.2f} GB" if n >= _MB_BELOW else f"{n / 1e6:.2f} MB"


def _plan(fetch: Fetch, data_fetch: Fetch, connectors: Sequence[str], data: Sequence[str]) -> list[str]:
    """``--dry-run``: one line per selected bundle with its size, then the total; nothing is installed."""
    lines, total = [], 0
    for kind, names, source in (("connector", connectors, fetch), ("data", data, data_fetch)):
        if not names:
            continue
        index_file, key = INDEXES[kind]
        entries = {e["name"]: e for e in json.loads(source(index_file).read_text())[key]}
        for name in names:
            if name not in entries:
                raise KeyError(f"no {kind} bundle {name!r} in {index_file}; it lists {sorted(entries)}")
            total += int(entries[name].get("bytes", 0))
            lines.append(f"would download {kind:<9}  {name:<36}  {_size(entries[name]):>9}")
    lines.append(f"total {len(lines)} bundles, {total / 1e9:.2f} GB (nothing downloaded: --dry-run)")
    return lines


def connector_kind_text(bundle: dict[str, Any]) -> str:
    """Return a connector bundle's kind, input (pooled or residues), token count and DEMO marker, for ``--kinds``.

    Read from ``bundle.json`` alone: its ``connector_kind`` when it has one, else ``write`` (the kind and tokens) and
    the kind's registered input (a plugin kind's input is printed as unknown: its plugin is not imported here).
    """
    from latent_interconnect.plugins import UnknownNameError  # noqa: PLC0415
    from latent_interconnect.writes import MAP_KINDS  # noqa: PLC0415  (pulls in torch: only for --kinds)

    layout = bundle.get("connector_kind")
    if layout is not None:
        kind, reads, tokens = layout["kind"], layout["input"], int(layout["tokens"])
    else:
        write = bundle["write"]
        kind, tokens = str(write["kind"]), len(write["tokens"])
        try:
            reads = MAP_KINDS.get(kind).input_kind.value
        except UnknownNameError:
            reads = "input unknown (plugin kind)"
    return f"{kind}, {reads}, {tokens} token(s){', DEMO' if bundle.get('demo') else ''}"


def list_bundles(fetch: Fetch, data_fetch: Fetch, *, kinds: bool = False) -> list[str]:
    """One line per bundle of both indexes: kind, name, size (:func:`_size`) and the description, if any.

    With ``kinds``, each connector's line also gives :func:`connector_kind_text`, from its ``bundle.json`` (one small
    file per connector, checked against the index's sha256; the index itself lists no kinds).
    """
    lines = []
    for kind, source in (("connector", fetch), ("data", data_fetch)):
        index_file, key = INDEXES[kind]
        for entry in json.loads(source(index_file).read_text())[key]:
            size = _size(entry)
            desc = f"  {entry['description']}" if entry.get("description") else ""
            what = (
                f"  [{connector_kind_text(_bundle(source, kind, entry['name'])[1])}]"
                if kinds and kind == "connector"
                else ""
            )
            lines.append(f"{kind:<9}  {entry['name']:<36}  {size:>8}{what}{desc}")
    return lines


def _integrity_message(err: BundleIntegrityError, args: argparse.Namespace) -> str:
    """One line for a failed check: the file, and where the damaged copy is to delete before a rerun."""
    if err.installed:
        where = f"the installed copy in {args.dest}/lip/artifacts is damaged: delete that artifact's directory"
    elif args.local_source is not None:
        where = f"the copy under {args.local_source} differs from the release"
    else:
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415  (the optional `hf` extra)

        repo, prefix = (args.repo, "models") if err.kind == "connector" else (args.data_repo, "datasets")
        entry = Path(HF_HUB_CACHE) / f"{prefix}--{repo.replace('/', '--')}"
        where = f"the cached download may be damaged: delete the Hugging Face cache entry {entry}"
    return f"error: {err.kind} bundle {err.name}: {err}; {where} and rerun\n"


def _revision(
    parser: argparse.ArgumentParser, repo: str, revision: str | None, *, release: str, commit: str, flag: str
) -> str:
    """Return the revision to read ``repo`` at: the given one, else the release commit for the release repository.

    A repository other than the release's must be read at a full commit sha (40 hex digits), so what is installed is
    fixed: a branch or tag such as ``main`` is refused.
    """
    if repo != release:
        if revision is None:
            parser.error(f"{repo} is not the release repository: give {flag} <commit> (the one its listing names)")
        if _COMMIT.fullmatch(revision) is None:
            parser.error(
                f"{flag} {revision!r} is not a commit: {repo} is not the release repository, so give the full "
                "40-hex commit sha its listing names (a branch or tag can move)"
            )
        return revision
    return commit if revision is None else revision


def main(argv: Sequence[str] | None = None) -> int:
    """The command line (module docstring)."""
    parser = argparse.ArgumentParser(prog="python -m latent_interconnect.release_download", description=__doc__)
    parser.add_argument(
        "--repo",
        default=CONNECTORS_REPO,
        help=f"the connectors' model repository ({CONNECTORS_REPO}, or a contributor's)",
    )
    parser.add_argument(
        "--revision", help="its revision (default: the release commit; a full commit sha, required for another --repo)"
    )
    parser.add_argument("--data-repo", default=DATA_REPO, help=f"the data bundles' dataset repository ({DATA_REPO})")
    parser.add_argument(
        "--data-revision",
        help="its revision (default: the release commit; a full commit sha, required for another repo)",
    )
    parser.add_argument("--connector", action="append", default=[], help="a connector bundle name (repeatable)")
    parser.add_argument("--data", action="append", default=[], help="a data bundle name (repeatable)")
    parser.add_argument("--dest", type=Path, help="the data root to install into (LIP_DATA; required unless --list)")
    parser.add_argument(
        "--all-data", action="store_true", help="every data bundle of the data index except DEMO bundles (about 30 GB)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the selected bundles, their sizes and the total; install nothing"
    )
    parser.add_argument("--list", action="store_true", help="list every bundle with its size and exit")
    parser.add_argument(
        "--kinds",
        action="store_true",
        help="with --list: also each connector's kind, input, token count and DEMO marker (reads each bundle.json)",
    )
    parser.add_argument(
        "--local-source",
        type=Path,
        help="read DIR/connectors and DIR/data instead of the hub (for checking a local bundle)",
    )
    args = parser.parse_args(argv)
    if args.kinds and not args.list:
        parser.error("--kinds goes with --list")
    if not args.list and not args.connector and not args.data and not args.all_data:
        parser.error("name at least one --connector or --data bundle, or --all-data (or --list)")
    if not args.list and not args.dry_run and args.dest is None:
        parser.error("the following arguments are required: --dest")
    if args.local_source is not None:
        fetch, data_fetch = local_fetch(args.local_source / "connectors"), local_fetch(args.local_source / "data")
    else:
        revision = _revision(
            parser, args.repo, args.revision, release=CONNECTORS_REPO, commit=CONNECTORS_REVISION, flag="--revision"
        )
        fetch = hub_fetch(args.repo, revision, "model")
        data_fetch = fetch  # unused without --data, --all-data or --list: the data repository is then not read
        if args.data or args.all_data or args.list:
            data_revision = _revision(
                parser,
                args.data_repo,
                args.data_revision,
                release=DATA_REPO,
                commit=DATA_REVISION,
                flag="--data-revision",
            )
            data_fetch = hub_fetch(args.data_repo, data_revision, "dataset")
    if args.list:
        sys.stdout.write("".join(line + "\n" for line in list_bundles(fetch, data_fetch, kinds=args.kinds)))
        return 0
    data = list(args.data)
    if args.all_data:
        data += [n for n in all_data_names(data_fetch) if n not in data]
    if args.dry_run:
        sys.stdout.write("".join(line + "\n" for line in _plan(fetch, data_fetch, args.connector, data)))
        return 0
    report: list[str] = []
    try:
        pins, verified = _download(fetch, args.dest, args.connector, data, data_fetch, report=report)
    except BundleIntegrityError as err:
        sys.stderr.write(_integrity_message(err, args))
        return 1
    for name, pin in sorted(pins.items()):
        sys.stdout.write(f"pinned {name} -> {pin.target} {pin.sha256[:12]}\n")
    for line in report:
        sys.stdout.write(line + "\n")
    sys.stdout.write(f"verified {verified} installed files (sha256)\n")
    sys.stdout.write(f"export LIP_DATA={args.dest.resolve()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
