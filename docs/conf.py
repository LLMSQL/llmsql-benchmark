# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = "LLMSQL"
copyright = "2025, Dzmitry Pihulski"
author = "Viktoria Novogrodskaia"
release = "0.2.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

templates_path = []

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Статика (CSS/JS)
html_static_path = ["_static"]

# ❌ Не используем тему, чтобы не ломала верстку
html_theme = None

# ❗ Просто скопировать index.html как есть
html_extra_path = ["_templates"]