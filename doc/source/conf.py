# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Hopes"
copyright = "2024, EnergyWise"
author = "Antoine Galataud"

master_doc = "index"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.append(os.path.abspath("./_ext"))
sys.path.insert(0, os.path.abspath("../../"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    # responsive design
    "sphinx_design",
    # requires non-shallow clone, breaks CI
    # "sphinx_last_updated_by_git",
]

autodoc_mock_imports = [
    "sklearn",
    "torch",
    "pwlf",
    "onnxruntime",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"


def find_version():
    with open("../../pyproject.toml") as f:
        for line in f:
            if "version" in line:
                return line.split("=")[1].strip().replace('"', "")


release = find_version()

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_logo = "_static/img/logo.svg"

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = f"Hopes v{release}"

html_theme_options = {
    "use_edit_page_button": False,
    "announcement": None,
    "logo": {
        "alt_text": f"{html_title}",
        "text": f"{html_title}",
        "image_light": "_static/img/logo.svg",
        "image_dark": "_static/img/logo.svg",
    },
    "navbar_align": "content",
    "navigation_depth": 2,
    "footer_start": ["copyright"],
    "footer_end": ["footer-end"],
    "pygment_light_style": "stata-dark",
    "pygment_dark_style": "stata-dark",
}

html_sidebars = {
    "**": ["main-sidebar"],
}

html_context = {
    "github_user": "airboxlab",
    "github_repo": "hopes",
    "github_version": "master",
    "doc_path": "doc/source/",
}
