"""Homology searches as a declared step: MMseqs2 ``easy-search`` between two sequence sets, with recorded parameters.

A search (:class:`HomologySearchSpec`) names its query and target sequence sets, every MMseqs2 parameter that changes
its hits (:class:`MMseqsSearch`, the program's version included) and whether it also searches the other way. The
``homology`` stage (``latent_interconnect.run.homology``) runs it and commits a :data:`HOMOLOGY_KIND` artifact:

- ``hits.parquet``: one row per hit, :data:`HIT_COLUMNS` plus ``direction``. Every row reads in the spec's orientation:
  ``query`` is from the query set and ``target`` from the target set. A ``reverse`` hit (targets searched against the
  queries) is turned around, its ``qcov`` and ``tcov`` swapped with its ids, so ``qcov`` is always the coverage of the
  query-set sequence. Rows are sorted, so the artifact does not depend on MMseqs2's thread scheduling.
- the manifest metadata: the spec and its digest, the version the binary reported, the exact search arguments, the
  sha256 of each side's FASTA and the counts of sequences and hits.

A sequence set is either the proteins of a domain in some splits (:class:`DomainSequences`; ``Domain.population``,
so no derived object such as a variant) or rows of declared table files (:class:`TableSequences`, each file cited by
its sha256). FASTA files list the sequences in id order: MMseqs2's hits depend on the target order (prefilter ties
at ``--max-seqs``; measured by review V-TSE), so the order is part of the search. The binary's version must equal
the spec's (:class:`VersionMismatchError`), so a result cannot come from another release. Where the binary lives,
how many threads it uses and where its scratch goes are execution options of the stage, never identity (threads do
not change the hits: V-TSE).
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore, sha256_file
from latent_interconnect.config import Config, StageConfig
from latent_interconnect.domains.base import DomainSpec, open_domain
from latent_interconnect.domains.tables import SourceFile, read_source

__all__ = [
    "HIT_COLUMNS",
    "HOMOLOGY_KIND",
    "DomainSequences",
    "HomologySearchSpec",
    "MMseqsSearch",
    "SearchFailedError",
    "SequenceTable",
    "TableSequences",
    "VersionMismatchError",
    "load_hits",
    "mmseqs_version",
    "search_homologs",
    "sequences",
]

HOMOLOGY_KIND = "homology"
"""Artifact kind of a homology search's hits."""
HIT_COLUMNS = ("query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits")
"""The hit columns, MMseqs2's ``--format-output`` names, in this order."""
_HITS_FILE = "hits.parquet"
_DIRECTION = "direction"
_COV_MODES = 5  # MMseqs2 --cov-mode takes 0..5
_LOG_TAIL = 20  # lines of MMseqs2's output quoted when it fails


class VersionMismatchError(RuntimeError):
    """The MMseqs2 binary reports another version than the search spec declares."""


class SearchFailedError(RuntimeError):
    """MMseqs2 exited with an error (the message quotes the end of its output)."""


class MMseqsSearch(Config):
    """The MMseqs2 ``easy-search`` parameters this project varies, each stated.

    Every other parameter is the pinned version's default (alignment mode, ``--seq-id-mode``, which defines
    ``fident``, the composition-bias correction, masking, the k-mer size, ...). One default is not fixed by the
    version: ``--split``, which MMseqs2 chooses from the memory it finds; it is not recorded.

    Attributes:
        version: the MMseqs2 version as ``mmseqs version`` prints it (a release tag or a commit hash); the binary
            must report exactly this.
        sensitivity: ``-s`` (MMseqs2's default is 5.7).
        evalue: ``-e``, the largest E-value kept.
        max_seqs: ``--max-seqs``, the prefilter's candidates per query.
        min_seq_id: ``--min-seq-id``, the smallest sequence identity kept (0: no cut).
        coverage: ``-c``, the smallest coverage kept (0: no cut).
        cov_mode: ``--cov-mode`` (0: the coverage of both sequences).
    """

    version: str = Field(min_length=1)
    sensitivity: float = Field(gt=0)
    evalue: float = Field(gt=0)
    max_seqs: int = Field(ge=1)
    min_seq_id: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    cov_mode: int = Field(ge=0, le=_COV_MODES)

    def arguments(self) -> list[str]:
        """Return the search arguments, in a fixed order (recorded in the artifact)."""
        return [
            "-s",
            repr(self.sensitivity),
            "-e",
            repr(self.evalue),
            "--max-seqs",
            str(self.max_seqs),
            "--min-seq-id",
            repr(self.min_seq_id),
            "-c",
            repr(self.coverage),
            "--cov-mode",
            str(self.cov_mode),
            "--format-output",
            ",".join(HIT_COLUMNS),
        ]


class DomainSequences(Config):
    """The proteins of a domain in some splits: ``Domain.population`` (no derived objects), its ``value`` as sequence.

    Attributes:
        domain: the domain at a fixed data version.
        splits: the splits whose objects are searched.
    """

    domain: DomainSpec
    splits: tuple[Literal["train", "val", "test"], ...] = Field(min_length=1)


class SequenceTable(Config):
    """Sequences from one table file (parquet), one row per sequence.

    Attributes:
        table: the file, under the data root, with its sha256.
        id: the id column (ids contain no whitespace: MMseqs2 cuts a FASTA header there).
        sequence: the sequence column.
    """

    table: SourceFile
    id: str
    sequence: str


class TableSequences(Config):
    """Sequences from table files; ids are unique across them.

    Attributes:
        tables: the files.
    """

    tables: tuple[SequenceTable, ...] = Field(min_length=1)


class HomologySearchSpec(StageConfig):
    """One homology search: two sequence sets and the MMseqs2 parameters.

    Attributes:
        name: what the search is, for logs and the artifact's metadata.
        queries: the query set.
        targets: the target set.
        search: the MMseqs2 parameters and version.
        both_directions: also search the targets against the queries (a hit either way counts; the reverse hits
            are turned around, see the module docstring). Searching both ways finds more hits, since the prefilter
            and the E-value depend on which side is the database.
    """

    name: str = Field(min_length=1)
    queries: DomainSequences | TableSequences
    targets: DomainSequences | TableSequences
    search: MMseqsSearch
    both_directions: bool

    @model_validator(mode="after")
    def _check(self) -> HomologySearchSpec:
        for side in (self.queries, self.targets):
            if isinstance(side, DomainSequences) and len(set(side.splits)) != len(side.splits):
                raise ValueError(f"search {self.name!r}: splits repeat in {side.splits}")
        return self


def mmseqs_version(mmseqs: Path) -> str:
    """Return the version the MMseqs2 binary reports (``mmseqs version``).

    Raises:
        subprocess.CalledProcessError: if the binary fails.
        FileNotFoundError: if there is no such binary.
    """
    out = subprocess.run([str(mmseqs), "version"], check=True, capture_output=True, text=True)
    return out.stdout.strip()


def sequences(side: DomainSequences | TableSequences, store: ArtifactStore, data_root: Path) -> pd.Series:
    """Return one sequence set: id → sequence, sorted by id.

    Raises:
        IntegrityError: if a table file's sha256 differs from its citation.
        ValueError: if an id repeats or contains whitespace, a sequence is empty, or a table lacks a column.
    """
    if isinstance(side, DomainSequences):
        domain = open_domain(side.domain, store)
        population = domain.population()
        split = domain.split().reindex(population)
        ids = population[split.isin(side.splits).to_numpy()]
        out = domain.objects()["value"].reindex(ids).astype(str)
    else:
        parts = []
        for t in side.tables:
            frame = pd.read_parquet(read_source(data_root, t.table))
            missing = sorted({t.id, t.sequence} - set(frame.columns))
            if missing:
                raise ValueError(f"{t.table.path}: no column(s) {missing}")
            parts.append(pd.Series(frame[t.sequence].astype(str).to_numpy(), index=frame[t.id].astype(str)))
        out = pd.concat(parts)
    out = out.rename_axis(None).rename(None)
    if not out.index.is_unique:
        raise ValueError(f"sequence ids repeat, e.g. {list(out.index[out.index.duplicated()][:3])}")
    spaced = out.index[out.index.str.contains(r"\s", regex=True)]
    if len(spaced):
        raise ValueError(f"sequence ids contain whitespace, e.g. {list(spaced[:3])}")
    if (out.str.len() == 0).any():
        raise ValueError(f"empty sequences, e.g. {list(out.index[out.str.len() == 0][:3])}")
    return out.sort_index()


def search_homologs(
    spec: HomologySearchSpec,
    store: ArtifactStore,
    data_root: Path,
    *,
    mmseqs: Path,
    threads: int | None,
    workdir: Path,
) -> ArtifactRef:
    """Run the search and commit its :data:`HOMOLOGY_KIND` artifact (the module docstring gives the format).

    A search already committed is returned as it is, never recomputed (``ArtifactStore.committed``). The FASTA files
    and MMseqs2's scratch go into a new directory under ``workdir``, removed when the search ends.

    Args:
        spec: the search.
        store: the artifact store.
        data_root: the data root that table files are under (``Paths.data``).
        mmseqs: the MMseqs2 binary.
        threads: MMseqs2 ``--threads`` (None: its default, every core).
        workdir: where the scratch directory is made (local disk is much faster than a network filesystem).

    Raises:
        VersionMismatchError: if the binary reports another version than ``spec.search.version``.
        ValueError: as :func:`sequences`, or if MMseqs2 reports an id that is not in its set.
        SearchFailedError: if MMseqs2 fails.
    """
    key = f"homology-{spec.digest()}"
    done = store.committed(key)
    if done is not None:
        return done
    found = mmseqs_version(mmseqs)
    if found != spec.search.version:
        raise VersionMismatchError(
            f"{mmseqs} reports MMseqs2 {found!r}; search {spec.name!r} declares {spec.search.version!r}"
        )
    queries = sequences(spec.queries, store, data_root)
    targets = sequences(spec.targets, store, data_root)
    scratch = Path(workdir) / f"homology-{uuid.uuid4().hex}"
    scratch.mkdir(parents=True)
    try:
        q_fasta, t_fasta = _fasta(scratch / "queries.fasta", queries), _fasta(scratch / "targets.fasta", targets)
        runs = [("forward", q_fasta, t_fasta)]
        if spec.both_directions:
            runs.append(("reverse", t_fasta, q_fasta))
        parts = []
        for direction, a, b in runs:
            hits = _easy_search(mmseqs, a, b, scratch / direction, search=spec.search, threads=threads)
            if direction == "reverse":
                hits = hits.rename(columns={"query": "target", "target": "query", "qcov": "tcov", "tcov": "qcov"})
            parts.append(hits[list(HIT_COLUMNS)].assign(**{_DIRECTION: direction}))
        hits = pd.concat(parts, ignore_index=True)
        for column, known in (("query", queries.index), ("target", targets.index)):
            stray = hits[column][~hits[column].isin(known)]
            if len(stray):
                raise ValueError(
                    f"search {spec.name!r}: MMseqs2 reports {column} ids outside its set: {list(stray[:3])}"
                )
        hits = hits.sort_values([_DIRECTION, "query", "target", "evalue", "bits"], kind="stable", ignore_index=True)
        staged = store.staging(key)
        hits.to_parquet(staged / _HITS_FILE, index=False)
        meta = {
            "spec": spec.model_dump(mode="json"),
            "spec_digest": spec.digest(),
            "mmseqs_version": found,
            "arguments": spec.search.arguments(),
            "queries": {"n": len(queries), "fasta_sha256": sha256_file(q_fasta)},
            "targets": {"n": len(targets), "fasta_sha256": sha256_file(t_fasta)},
            "hits": {d: int((hits[_DIRECTION] == d).sum()) for d, _, _ in runs},
        }
        return store.commit(staged, HOMOLOGY_KIND, meta)
    finally:
        shutil.rmtree(scratch)


def load_hits(store: ArtifactStore, ref: ArtifactRef) -> pd.DataFrame:
    """Return a committed search's hits (:data:`HIT_COLUMNS` and ``direction``), its manifest checked.

    Raises:
        ValueError: if ``ref`` is not a :data:`HOMOLOGY_KIND` artifact.
    """
    if ref.kind != HOMOLOGY_KIND:
        raise ValueError(f"{ref.short} is a {ref.kind!r} artifact, not a homology search")
    store.manifest(ref)
    return pd.read_parquet(store.path(ref) / _HITS_FILE)


def _fasta(path: Path, seqs: pd.Series) -> Path:
    with path.open("w") as fh:
        for i, s in seqs.items():
            fh.write(f">{i}\n{s}\n")
    return path


def _easy_search(
    mmseqs: Path, queries: Path, targets: Path, run: Path, *, search: MMseqsSearch, threads: int | None
) -> pd.DataFrame:
    """Run one ``easy-search`` in ``run`` (made here) and return its hits (none: an empty frame)."""
    run.mkdir()
    out = run / "hits.m8"
    command = [str(mmseqs), "easy-search", str(queries), str(targets), str(out), str(run / "tmp"), *search.arguments()]
    if threads is not None:
        command += ["--threads", str(threads)]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        tail = "\n".join((done.stdout + done.stderr).splitlines()[-_LOG_TAIL:])
        raise SearchFailedError(f"MMseqs2 easy-search exited {done.returncode}:\n{tail}")
    columns = list(HIT_COLUMNS)
    if out.stat().st_size == 0:
        return pd.DataFrame({c: pd.Series(dtype=str if c in ("query", "target") else float) for c in columns})
    dtypes = {c: str if c in ("query", "target") else float for c in columns}
    return pd.read_csv(out, sep="\t", header=None, names=columns, dtype=dtypes)
