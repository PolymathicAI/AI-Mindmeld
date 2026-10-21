"""The molecule text renderer: the paper's ChEBI molecule text, rendered for any molecule from declared sources.

A faithful port of the paper's molecule text builders (its templates and its field table), whose output covers only the
molecules of the first molecule split. The content rules are unchanged; each module states the original code it ports:

- ``definitions``: ChEBI text cleaning, names and the usable definition (the release is read by ``domains.chebi``);
- ``roles``: the B4 role split and a molecule's trained and held-out roles;
- ``classes``: the partition of the parent classes into the two class fields;
- ``sentences``: sentence splitting, the connective rule, the own-name rewrite (with its organism guard)
  and the sentence-boundary trim;
- ``fields``: one molecule's fields, with the item rule of ``domains.eval_proteins`` (no evaluation protein, no EC
  number);
- ``table``: the table over a split's molecules (:class:`MoleculeTextSpec`), its checks and its artifact.

The ``molecule_text`` stage (``latent_interconnect.run.molecule_text``) builds and commits a table. The molecule
domain reads a table as its ``text`` source (``domains.molecule.MoleculeDataSpec``); the paper's table and a table this
package built have the same columns, so either is a config change.
"""

from latent_interconnect.domains.molecule_text.table import (
    CHECKS,
    COLUMNS,
    MOLECULE_TEXT_KIND,
    TEXT_FILE,
    MoleculeTextSpec,
    TextTable,
    build_text_table,
    check_text_table,
    clear_held_out,
    commit_text_table,
)

__all__ = [
    "CHECKS",
    "COLUMNS",
    "MOLECULE_TEXT_KIND",
    "TEXT_FILE",
    "MoleculeTextSpec",
    "TextTable",
    "build_text_table",
    "check_text_table",
    "clear_held_out",
    "commit_text_table",
]
