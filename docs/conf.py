# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import llmsql

project = "LLMSQL"
release = llmsql.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True


autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_static_path = ["_static"]

html_favicon = "_static/logo.jpg"

html_theme = "basic"

html_show_sphinx = False
html_show_copyright = False


html_additional_pages = {
    "index": "index.html",
}

pygments_style = "monokai"
pygments_dark_style = "monokai"
highlight_language = "python"


def setup(app):
    app.add_css_file("styles/front_page.css")
    app.add_js_file("scripts/front_page.js")
