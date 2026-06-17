import pytest

pytestmark = pytest.mark.skip(reason="scaffold only")


def test_package_exports_core_public_functions():
    """Test public imports from jaxincell.

    Cases to implement:
    - Simulation and load_parameters are exported.
    - diagnostics and plot are exported.
    - field, particle, source, filter, and boundary-condition helpers expected by examples remain importable.
    """


def test_package_import_does_not_mask_module_exports():
    """Test jaxincell package import behavior.

    Cases to implement:
    - wildcard imports in jaxincell.__init__ do not mask core public names.
    - importing jaxincell does not require optional plotting backends beyond current package requirements.
    - repeated imports are idempotent.
    """


def test_version_metadata_is_importable():
    """Test package metadata access.

    Cases to implement:
    - jaxincell.version exposes version and __version__ from setuptools_scm output.
    - importlib.metadata.version("jaxincell") works when the package is installed.
    - local editable/import fallback behavior is documented by the test.
    """
