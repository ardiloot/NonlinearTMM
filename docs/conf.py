from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "numpydoc",
]

source_suffix = ".rst"
root_doc = "index"

project = "NonlinearTMM"
copyright = "2017-2026, Ardi Loot"
author = "Ardi Loot"

__version__ = importlib.metadata.version("NonlinearTMM")
version = __version__
release = __version__

language = "en"
exclude_patterns = ["_build", "_autosummary", "Thumbs.db", ".DS_Store"]
todo_include_todos = True

# Suppress :any: cross-reference warnings from numpydoc — these are parameter
# names (e.g. paramStr, layerNr) that numpydoc converts to :any: roles but
# that have no matching Sphinx targets.
suppress_warnings = ["ref.any"]

# -- autosummary -----------------------------------------------------------

autosummary_generate = True

# -- numpydoc --------------------------------------------------------------

numpydoc_xref_param_type = False
numpydoc_show_class_members = False

# -- intersphinx -----------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# -- Options for HTML output ----------------------------------------------

html_theme = "furo"
html_theme_options = {
    "navigation_with_keys": True,
}


# -- Options for HTMLHelp output ------------------------------------------

htmlhelp_basename = "NonlinearTMMdoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}

latex_documents = [
    (root_doc, "NonlinearTMM.tex", "NonlinearTMM Documentation", "Ardi Loot", "manual"),
]


# -- Options for manual page output ---------------------------------------

man_pages = [(root_doc, "nonlineartmm", "NonlinearTMM Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

texinfo_documents = [
    (
        root_doc,
        "NonlinearTMM",
        "NonlinearTMM Documentation",
        author,
        "NonlinearTMM",
        "Nonlinear transfer-matrix method.",
        "Miscellaneous",
    ),
]
