"""Sphinx configuration for the JAX-in-Cell documentation."""
import datetime
import json
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS.parent))

project = "JAX-in-Cell"
author = "UWPlasma, University of Wisconsin-Madison"
copyright = f"{datetime.date.today().year}, UWPlasma"

try:
    from importlib.metadata import version as _version
    release = _version("jaxincell")
except Exception:  # building without the package installed
    release = "dev"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "scripts/*"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- MyST ------------------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
myst_dmath_double_inline = True

# Numbers quoted in the text come from the same runs that made the figures.
_measurements = DOCS / "_static" / "figures" / "measurements.json"


def _format(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        magnitude = abs(value)
        if magnitude < 1e-3 or magnitude >= 1e4:
            return f"{value:.2e}"
        return f"{value:.3g}"
    if isinstance(value, list):
        return ", ".join(_format(v) for v in value)
    return str(value)


myst_substitutions = {}
if _measurements.exists():
    for _key, _value in json.loads(_measurements.read_text()).items():
        if isinstance(_value, dict):
            continue
        myst_substitutions[_key] = _format(_value)

# -- autodoc / autosummary --------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- Bibliography -------------------------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# -- Copy button ----------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- HTML -------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "JAX-in-Cell"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/JAX-in-Cell_icon.png"
html_show_sourcelink = False
html_theme_options = {
    "logo": {
        "image_light": "_static/JAX-in-Cell_logo.png",
        "image_dark": "_static/JAX-in-Cell_logo_dark.png",
        "alt_text": "JAX-in-Cell",
    },
    "github_url": "https://github.com/uwplasma/JAX-in-Cell",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/jaxincell/",
            "icon": "fa-brands fa-python",
        },
    ],
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "show_prev_next": True,
    "footer_start": ["copyright"],
    "footer_end": [],
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
}
html_context = {
    "github_user": "uwplasma",
    "github_repo": "JAX-in-Cell",
    "github_version": "main",
    "doc_path": "docs",
}
html_sidebars = {
    "index": [],
}
