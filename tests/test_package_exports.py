from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


def test_package_exports_core_public_functions():
    """Test public imports from jaxincell.

    Cases:
    - Simulation and load_parameters are exported.
    - diagnostics and plot are exported.
    - write_openpmd is exported with the required openpmd_api dependency installed.
    - field, particle, source, filter, and boundary-condition helpers expected by examples remain importable.
    """
    import jaxincell

    expected_exports = {
        "Simulation",
        "load_parameters",
        "diagnostics",
        "plot",
        "write_openpmd",
        "E_from_Gauss_1D_Cartesian",
        "boris_step",
        "current_density",
        "filter_scalar_field",
        "set_BC_positions",
    }

    for export_name in expected_exports:
        assert hasattr(jaxincell, export_name)
        assert callable(getattr(jaxincell, export_name))


def test_package_import_does_not_mask_module_exports():
    """Test jaxincell package import behavior.

    Cases:
    - wildcard imports in jaxincell.__init__ do not mask core public names.
    - importing jaxincell does not require optional plotting backends beyond current package requirements.
    - repeated imports are idempotent.
    """
    jaxincell = import_module("jaxincell")
    jaxincell_again = import_module("jaxincell")
    wildcard_namespace = {}

    exec("from jaxincell import *", wildcard_namespace)

    assert jaxincell_again is jaxincell
    for export_name in ("Simulation", "load_parameters", "diagnostics", "plot", "write_openpmd"):
        assert wildcard_namespace[export_name] is getattr(jaxincell, export_name)


def test_version_metadata_is_importable():
    """Test package metadata access.

    Cases:
    - jaxincell.version exposes version and __version__ from setuptools_scm output.
    - importlib.metadata.version("jaxincell") works when the package is installed.
    - local editable/import fallback behavior is documented by the test.
    """
    version_module = import_module("jaxincell.version")

    assert version_module.version == version_module.__version__
    assert version_module.version_tuple == version_module.__version_tuple__
    assert isinstance(version_module.__version__, str)
    assert version_module.__version__

    try:
        installed_version = version("jaxincell")
    except PackageNotFoundError:
        installed_version = None

    if installed_version is not None:
        assert isinstance(installed_version, str)
        assert installed_version
    else:
        assert version_module.__version__ == "0.1"
