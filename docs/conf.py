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
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "scripts/*"]

myst_enable_extensions = ["amsmath", "colon_fence", "dollarmath", "substitution"]
myst_heading_anchors = 3
myst_dmath_double_inline = True


def _format(value):
    """%g keeps the precision the scripts rounded to (1.4041, not 1.4)."""
    if isinstance(value, list):
        return ", ".join(_format(v) for v in value)
    return f"{value:g}" if isinstance(value, float) else str(value)


# Numbers quoted in the text come from the same runs that made the figures.
myst_substitutions = {key: _format(value) for key, value in
                      json.loads((DOCS / "_static" / "figures" / "measurements.json").read_text()).items()
                      if not isinstance(value, dict)}

autodoc_default_options = {"members": True}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_use_rtype = False

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

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
    "icon_links": [{"name": "PyPI", "url": "https://pypi.org/project/jaxincell/", "icon": "fa-brands fa-python"}],
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": [],
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
}
html_context = {"github_user": "uwplasma", "github_repo": "JAX-in-Cell", "github_version": "main", "doc_path": "docs"}
html_sidebars = {"index": []}
