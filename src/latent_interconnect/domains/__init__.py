"""Object domains: the contract, the domain-data format and the views training streams read.

``base`` holds the contract, ``tables`` the domain-data format and its build specs, and ``views`` the restricted views
training streams read. ``chebi`` is the one ChEBI release reader, ``eval_proteins`` the molecule text's
evaluation-protein item rule, and ``molecule_text`` the molecule text renderer.

``protein``, ``molecule`` and ``pfmbench`` (PFMbench's test proteins) are registered in ``DOMAINS`` (and their data
specs in ``DOMAIN_DATA``) on import.
"""

from latent_interconnect.domains.base import (
    DOMAIN_DATA_KIND,
    DOMAINS,
    QUESTION_MARKER,
    CaptureRecipe,
    Domain,
    DomainSpec,
    Frame,
    RecipeMessage,
    TaskKind,
    TaskScope,
    TaskSpec,
    open_domain,
    require_splits,
)
from latent_interconnect.domains.molecule import MoleculeDomain
from latent_interconnect.domains.pfmbench import PfmbenchDomain
from latent_interconnect.domains.protein import ProteinDomain
from latent_interconnect.domains.tables import DOMAIN_DATA, DomainDataSpec, SourceFile, TableDomain
from latent_interconnect.domains.views import CatalogView, DomainCatalog, DomainView, RestrictedDomain

__all__ = [
    "DOMAINS",
    "DOMAIN_DATA",
    "DOMAIN_DATA_KIND",
    "QUESTION_MARKER",
    "CaptureRecipe",
    "CatalogView",
    "Domain",
    "DomainCatalog",
    "DomainDataSpec",
    "DomainSpec",
    "DomainView",
    "Frame",
    "MoleculeDomain",
    "PfmbenchDomain",
    "ProteinDomain",
    "RecipeMessage",
    "RestrictedDomain",
    "SourceFile",
    "TableDomain",
    "TaskKind",
    "TaskScope",
    "TaskSpec",
    "open_domain",
    "require_splits",
]
