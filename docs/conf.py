# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = "LLMSQL"
copyright = "2025, Dzmitry Pihulski"
author = "Viktoria Novogrodskaia"
release = "0.2.0"

# -- General configuration ---------------------------------------------------

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_static_path = ["_static"]

html_theme = "basic"

html_additional_pages = {
    "index": "index.html",
}

def setup(app):
    app.add_css_file("styles/front_page.css")
    app.add_js_file("scripts/front_page.js")