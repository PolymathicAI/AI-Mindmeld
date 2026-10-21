"""Uni-Mol2's opt-in conformer retry (worker side): a second 3D embedding before unimol_tools' 2D fallback.

unimol_tools 0.1.6 builds a molecule's conformer in ``inner_smi2coords``: RDKit's ``EmbedMolecule`` with hydrogens,
seed 42, MMFF-optimized; when that embedding fails (mode ``fast``) it puts 2D coordinates (``Compute2DCoords``) in
their place, silently. The failures are mostly large, flexible molecules whose random-distance-geometry start fails
(``INITIAL_COORDS``), and flat coordinates are wrong for them.

:func:`conformer` runs unimol_tools' own ``inner_smi2coords`` first, unchanged. Only when it returns 2D coordinates
(its conformer is not 3D: a failed embedding) does it embed again, from random starting coordinates
(``ETKDGv3``, ``useRandomCoords``), with chirality enforced (never disabled), the same seed and a bounded time
(RDKit's own ``timeout``), then MMFF-optimizes as unimol_tools does when MMFF has the molecule's parameters. The
retried conformer is kept only if every stereocentre and double bond the SMILES specifies has the same label when
read from the 3D coordinates; otherwise, or when the retry fails, unimol_tools' 2D conformer is kept. A molecule that
embedded the first time, or that has no conformer at all (unimol_tools then raises ``Bad Conformer Id``), is
returned exactly as unimol_tools built it.

The retry depends on time: a molecule whose retry ends near the limit may embed on one machine and not on another.
The routes (:data:`FIRST_3D`, :data:`RETRY_3D`, :data:`FALLBACK_2D`) are returned so the extractor can record them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["FALLBACK_2D", "FIRST_3D", "RETRY_3D", "conformer", "stereo_labels"]

FIRST_3D = "first-try 3D"
"""unimol_tools' own conformer: its first embedding succeeded (or the molecule has no conformer at all)."""
RETRY_3D = "retry 3D: random starting coordinates, chirality enforced"
"""The retry's conformer."""
FALLBACK_2D = "2D fallback"
"""unimol_tools' 2D conformer, kept after the retry (the note names why)."""


def stereo_labels(mol: Any) -> dict[str, str]:
    """Return the specified stereo labels of a molecule's heavy-atom graph: ``a<i>`` → R/S, ``b<i>`` → E/Z.

    Atom and bond indices are the molecule's own; ``AddHs`` appends hydrogens after the heavy atoms and their bonds
    after the heavy-atom bonds, so the labels of a SMILES and of its hydrogen-added conformer compare index by index.
    Unspecified centers and bonds are left out.
    """
    from rdkit import Chem  # noqa: PLC0415 - the encoder environment's RDKit

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    labels = {f"a{a.GetIdx()}": a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")}
    for b in mol.GetBonds():
        if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            labels[f"b{b.GetIdx()}"] = str(b.GetStereo())
    return labels


def _labels_from_3d(mol: Any) -> dict[str, str]:
    from rdkit import Chem  # noqa: PLC0415

    copy = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(copy)
    return stereo_labels(copy)


def conformer(smiles: str, seed: int, mode: str, remove_hs: bool, timeout_s: int) -> tuple[bytes, str]:
    """Return a molecule's conformer, as unimol_tools builds it or retried, and its route.

    Args:
        smiles: the molecule.
        seed: unimol_tools' conformer seed (``UniMolV2Feature.seed``); the retry uses it too.
        mode: unimol_tools' conformer mode (``UniMolV2Feature.mode``).
        remove_hs: unimol_tools' ``remove_hs`` (``UniMolV2Feature.remove_hs``).
        timeout_s: the retry's time limit (RDKit's ``EmbedParameters.timeout``, seconds).

    Returns:
        The molecule with its hydrogens and conformer, pickled with double-precision coordinates (``Chem.Mol(bytes)``
        restores it exactly), and its route: :data:`FIRST_3D`, :data:`RETRY_3D` or :data:`FALLBACK_2D` followed by
        the reason.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415
    from unimol_tools.data.conformer import inner_smi2coords  # noqa: PLC0415

    def pickled(m: Any) -> bytes:
        return bytes(m.ToBinary(Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble))

    mol = inner_smi2coords(smiles, seed=seed, mode=mode, remove_hs=remove_hs, return_mol=True)
    if mol.GetNumConformers() == 0 or mol.GetConformer().Is3D():
        return pickled(mol), FIRST_3D
    retry = Chem.Mol(mol)
    retry.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    params.enforceChirality = True
    params.timeout = timeout_s
    if AllChem.EmbedMolecule(retry, params) != 0:
        return pickled(mol), f"{FALLBACK_2D}: the retry found no 3D embedding (limit {timeout_s} s)"
    if AllChem.MMFFHasAllMoleculeParams(retry):  # unimol_tools keeps the embedded coordinates when MMFF fails
        AllChem.MMFFOptimizeMolecule(retry)
    wanted = stereo_labels(Chem.MolFromSmiles(smiles))
    found = _labels_from_3d(retry)
    lost = sorted(k for k, v in wanted.items() if found.get(k) != v)
    if lost:
        return pickled(mol), f"{FALLBACK_2D}: the retried conformer changes the stereo of {lost[:5]}"
    return pickled(retry), RETRY_3D
