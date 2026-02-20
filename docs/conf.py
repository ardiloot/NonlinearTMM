from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

html_theme = "alabaster"
html_theme_options = {
    "page_width": "1200px",
    "sidebar_width": "300px",
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
