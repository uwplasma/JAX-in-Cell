import builtins
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from jaxincell import Simulation
from jaxincell._openpmd import (
    _open_series,
    _set_record_metadata,
    _store,
    _write_meshes,
    openpmd_output_paths,
    write_openpmd,
)
from jaxincell._parameters._export_parameters import clean_and_initialize_export_parameters
from jaxincell._parameters._solver_parameters import clean_and_initialize_solver_parameters
from tests.helpers import base_simulation_parameters

OPENPMD_EXTENSIONS = (".h5", ".bp", ".json")


class FakeDataset:
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = tuple(shape)


class FakeRecord:
    def __init__(self, name):
        self.name = name
        self.components = {}
        self.attributes = {}
        self.dataset = None
        self.data = None
        self.offset = None
        self.extent = None

    def __getitem__(self, key):
        self.components.setdefault(key, FakeRecord(f"{self.name}/{key}"))
        return self.components[key]

    def reset_dataset(self, dataset):
        self.dataset = dataset

    def store_chunk(self, data, offset, extent):
        self.data = np.asarray(data)
        self.offset = list(offset)
        self.extent = tuple(extent)

    def set_attribute(self, name, value):
        self.attributes[name] = value


class FakeGroup(FakeRecord):
    pass


class FakeMap:
    def __init__(self, factory):
        self.factory = factory
        self.entries = {}

    def __getitem__(self, key):
        self.entries.setdefault(key, self.factory(key))
        return self.entries[key]


class FakeIteration:
    def __init__(self, index):
        self.index = index
        self.meshes = FakeMap(lambda name: FakeGroup(name))
        self.particles = FakeMap(lambda name: FakeGroup(name))
        self.time = None
        self.dt = None
        self.time_unit_SI = None


class FakeSeries:
    created = []

    def __init__(self, path, access):
        self.path = path
        self.access = access
        self.attributes = {}
        self.iterations = FakeMap(lambda index: FakeIteration(index))
        self.flushed = False
        self.closed = False
        FakeSeries.created.append(self)

    def set_attribute(self, name, value):
        self.attributes[name] = value

    def set_iteration_encoding(self, value):
        self.iteration_encoding = value

    def set_iteration_format(self, value):
        self.iteration_format = value

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


class AttributeOnlySeries:
    created = []

    def __init__(self, path, access):
        self.path = path
        self.access = access
        AttributeOnlySeries.created.append(self)


class CompatibilitySeries:
    created = []

    def __init__(self, path, access):
        self.path = path
        self.access = access
        self.attributes = {}
        self.encoding_values = []
        self.format_values = []
        CompatibilitySeries.created.append(self)

    def set_attribute(self, name, value):
        self.attributes[name] = value

    def set_iteration_encoding(self, value):
        self.encoding_values.append(value)
        if value == "enum-file":
            raise TypeError("old bindings only accept raw strings")
        self.iteration_encoding = value

    def set_iteration_format(self, value):
        self.format_values.append(value)
        if len(self.format_values) == 1:
            raise TypeError("old bindings require the full path")
        self.iteration_format = value

    def set_meshes_path(self, value):
        raise TypeError("old bindings reject this setter")

    def set_particles_path(self, value):
        self.particles_path_setter_value = value

    @property
    def particles_path(self):
        return None

    @particles_path.setter
    def particles_path(self, value):
        raise AttributeError("read-only property")


class StrictMetadataRecord:
    def __init__(self):
        self.attributes = {}

    def __setattr__(self, name, value):
        if name in ("unit_dimension", "time_offset"):
            raise AttributeError(f"{name} is read-only")
        object.__setattr__(self, name, value)

    def set_attribute(self, name, value):
        self.attributes[name] = value


class StrictRecordComponent(FakeRecord):
    def __setattr__(self, name, value):
        if name == "unit_SI":
            raise AttributeError("unit_SI is read-only")
        object.__setattr__(self, name, value)


class StrictMesh(FakeGroup):
    def __setattr__(self, name, value):
        if name in ("grid_unit_SI", "unit_SI"):
            raise AttributeError(f"{name} is read-only")
        object.__setattr__(self, name, value)


def fake_openpmd_module():
    FakeSeries.created = []
    return SimpleNamespace(
        Series=FakeSeries,
        Access=SimpleNamespace(create="create"),
        Dataset=FakeDataset,
        Geometry=SimpleNamespace(cartesian="cartesian"),
        Data_Order=SimpleNamespace(C="C"),
        Mesh_Record_Component=SimpleNamespace(SCALAR="SCALAR"),
        Iteration_Encoding=SimpleNamespace(
            group_based="groupBased",
            file_based="fileBased",
        ),
    )


def test_openpmd_version_import_fallback(monkeypatch):
    """Test the fallback used if jaxincell.version cannot be imported."""
    import jaxincell._openpmd as openpmd_module

    real_import = builtins.__import__

    def fail_version_import(name, globals=None, locals=None, fromlist=(), level=0):
        package = None if globals is None else globals.get("__package__")
        if name == "version" and package == "jaxincell" and level == 1:
            raise ImportError("version unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_version_import)
    try:
        assert importlib.reload(openpmd_module).__version__ == "unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(openpmd_module)


def tiny_openpmd_output(tmp_path, **export_overrides):
    solver_parameters = clean_and_initialize_solver_parameters(
        {
            "print_info": False,
            "filter_passes": 0,
        }
    )
    export_parameters = clean_and_initialize_export_parameters(
        {
            "openpmd_filename": str(tmp_path / "jaxincell_output.h5"),
            **export_overrides,
        }
    )
    total_steps = 3
    number_grid_points = 4
    return {
        "positions": np.arange(total_steps * 4 * 3, dtype=float).reshape(total_steps, 4, 3),
        "velocities": np.full((total_steps, 4, 3), 2.0),
        "weights": np.array([[2.0], [2.0], [4.0], [4.0]]),
        "charges": np.array([[-2.0], [-2.0], [4.0], [4.0]]),
        "masses": np.array([[6.0], [6.0], [12.0], [12.0]]),
        "species_integer_index": np.array([0, 0, 1, 1], dtype=np.int32),
        "electric_field": np.ones((total_steps, number_grid_points, 3)),
        "magnetic_field": np.full((total_steps, number_grid_points, 3), 2.0),
        "current_density": np.full((total_steps, number_grid_points, 3), 3.0),
        "charge_density": np.full((total_steps, number_grid_points), 4.0),
        "external_electric_field": np.full((number_grid_points, 3), 5.0),
        "external_magnetic_field": np.full((number_grid_points, 3), 6.0),
        "total_steps": total_steps,
        "time_array": np.array([0.0, 0.1, 0.2]),
        "dt": 0.1,
        "dx": 0.25,
        "length": 1.0,
        "solver_parameters": solver_parameters,
        "export_parameters": export_parameters,
        "species_parameters": {
            "electrons": {
                "_electrons0": {"user_label": "electron beam"},
            },
            "ions": {
                "_ions0": {"user_label": "ions0"},
            },
        },
    }


@pytest.mark.parametrize("extension", OPENPMD_EXTENSIONS)
def test_openpmd_output_paths_iterate_without_overwrite(tmp_path, extension):
    """Test jaxincell._openpmd.openpmd_output_paths.

    Cases:
    - missing extensions default to .h5.
    - supported extensions are preserved.
    - an occupied combined destination receives an integer suffix.
    - separate mesh/particle destinations share one suffix.
    - unsupported filename extensions are rejected.
    """
    paths = openpmd_output_paths(
        str(tmp_path / f"series{extension}"),
        write_pmd_sidecar=True,
    )
    assert paths["data"]["combined"] == str(tmp_path / f"series{extension}")
    assert paths["sidecar"]["combined"] == str(tmp_path / "series.pmd")

    no_extension = tmp_path / "no_extension"
    default_paths = openpmd_output_paths(
        str(no_extension),
        write_pmd_sidecar=False,
    )
    assert default_paths["data"]["combined"] == str(tmp_path / "no_extension.h5")

    occupied = tmp_path / f"occupied{extension}"
    occupied.write_text("old data")
    iterated = openpmd_output_paths(str(occupied), write_pmd_sidecar=False)
    assert iterated["data"]["combined"] == str(tmp_path / f"occupied0{extension}")

    occupied_sidecar = tmp_path / "sidecar_only.pmd"
    occupied_sidecar.write_text("old sidecar data")
    sidecar_iterated = openpmd_output_paths(
        str(tmp_path / f"sidecar_only{extension}"),
        write_pmd_sidecar=False,
    )
    assert sidecar_iterated["data"]["combined"] == str(tmp_path / f"sidecar_only0{extension}")
    assert sidecar_iterated["sidecar"] == {}

    separate_mesh = tmp_path / f"split_meshes{extension}"
    separate_mesh.write_text("old mesh data")
    separate = openpmd_output_paths(
        str(tmp_path / f"split{extension}"),
        separate_particles_and_meshes=True,
        write_pmd_sidecar=True,
    )
    assert separate["data"]["meshes"] == str(tmp_path / f"split0_meshes{extension}")
    assert separate["data"]["particles"] == str(tmp_path / f"split0_particles{extension}")
    assert separate["sidecar"]["meshes"] == str(tmp_path / "split0_meshes.pmd")

    with pytest.raises(ValueError, match="openpmd_filename must end"):
        openpmd_output_paths(str(tmp_path / "bad.nc"))


@pytest.mark.parametrize("extension", OPENPMD_EXTENSIONS)
def test_openpmd_output_paths_file_based_templates(tmp_path, extension):
    """Test fileBased openPMD filename template handling.

    Cases:
    - fileBased output inserts a padded iteration token before the extension.
    - collision checks expand the token over requested iterations.
    - explicit iteration tokens and supported extensions are preserved.
    - separate mesh/particle templates share one collision suffix.
    """
    paths = openpmd_output_paths(
        str(tmp_path / f"filebased{extension}"),
        iteration_encoding="fileBased",
        iteration_indices=[0, 2],
    )
    assert paths["data"]["combined"] == str(tmp_path / f"filebased_%06T{extension}")
    assert paths["sidecar"]["combined"] == str(tmp_path / "filebased.pmd")

    concrete_file = tmp_path / f"filebased_000002{extension}"
    concrete_file.write_text("old data")
    iterated = openpmd_output_paths(
        str(tmp_path / f"filebased{extension}"),
        iteration_encoding="fileBased",
        iteration_indices=[0, 2],
    )
    assert iterated["data"]["combined"] == str(tmp_path / f"filebased0_%06T{extension}")
    assert iterated["sidecar"]["combined"] == str(tmp_path / "filebased0.pmd")

    explicit = openpmd_output_paths(
        str(tmp_path / f"explicit_%T{extension}"),
        iteration_encoding="fileBased",
        iteration_indices=[3],
        write_pmd_sidecar=False,
    )
    assert explicit["data"]["combined"] == str(tmp_path / f"explicit_%T{extension}")
    assert explicit["sidecar"] == {}

    separate_mesh = tmp_path / f"split_meshes_000000{extension}"
    separate_mesh.write_text("old mesh data")
    separate = openpmd_output_paths(
        str(tmp_path / f"split{extension}"),
        separate_particles_and_meshes=True,
        iteration_encoding="fileBased",
        iteration_indices=[0],
    )
    assert separate["data"]["meshes"] == str(tmp_path / f"split0_meshes_%06T{extension}")
    assert separate["data"]["particles"] == str(tmp_path / f"split0_particles_%06T{extension}")
    assert separate["sidecar"]["meshes"] == str(tmp_path / "split0_meshes.pmd")


def test_openpmd_output_paths_edge_cases(monkeypatch, tmp_path):
    """Test less common openPMD output path branches."""
    with pytest.raises(ValueError, match="iteration_encoding must be"):
        openpmd_output_paths(
            str(tmp_path / "series.h5"),
            iteration_encoding="iterationBased",
        )

    monkeypatch.chdir(tmp_path)
    token_only = openpmd_output_paths(
        "%T.h5",
        iteration_encoding="fileBased",
        write_pmd_sidecar=False,
    )
    assert token_only["data"]["combined"] == "%T_%T.h5"

    class OneShotPattern:
        def __init__(self):
            self.search_calls = 0

        def search(self, text):
            self.search_calls += 1
            if self.search_calls == 1:
                return SimpleNamespace(group=lambda index: "%T")
            return None

        def sub(self, replacement, text):
            return "series"

    monkeypatch.setattr("jaxincell._openpmd.ITERATION_TOKEN_PATTERN", OneShotPattern())
    paths = openpmd_output_paths(
        "ignored.h5",
        iteration_encoding="fileBased",
        write_pmd_sidecar=False,
    )
    assert paths["data"]["combined"] == "series_%T.h5"


def test_open_series_compatibility_fallbacks(tmp_path):
    """Test compatibility paths for older or restricted openpmd_api objects."""

    class FailingSeries:
        def __init__(self, path, access):
            raise OSError("backend unavailable")

    failing_io = SimpleNamespace(
        Series=FailingSeries,
        Access=SimpleNamespace(create="create"),
    )
    with pytest.raises(RuntimeError, match="Could not create openPMD .bp series"):
        _open_series(str(tmp_path / "broken.bp"), failing_io, "groupBased")

    AttributeOnlySeries.created = []
    attribute_io = SimpleNamespace(
        Series=AttributeOnlySeries,
        Access=SimpleNamespace(create="create"),
    )
    group_series = _open_series(
        "attribute_only.h5",
        attribute_io,
        "groupBased",
        meshes_path="meshes",
    )
    assert group_series.iterationEncoding == "groupBased"
    assert group_series.iterationFormat == "/data/%T/"
    assert group_series.meshesPath == "meshes/"

    file_series = _open_series(
        "attribute_%T.h5",
        attribute_io,
        "fileBased",
    )
    assert file_series.iterationEncoding == "fileBased"
    assert file_series.iterationFormat == "attribute_%T.h5"

    CompatibilitySeries.created = []
    compatibility_io = SimpleNamespace(
        Series=CompatibilitySeries,
        Access=SimpleNamespace(create="create"),
        Iteration_Encoding=SimpleNamespace(file_based="enum-file"),
    )
    compatibility_path = tmp_path / "compat_%06T.h5"
    compatibility_series = _open_series(
        str(compatibility_path),
        compatibility_io,
        "fileBased",
        meshes_path="meshes",
        particles_path="particles",
    )
    assert compatibility_series.encoding_values == ["enum-file", "fileBased"]
    assert compatibility_series.format_values == [
        compatibility_path.name,
        str(compatibility_path),
    ]
    assert compatibility_series.iteration_format == str(compatibility_path)
    assert compatibility_series.meshes_path == "meshes/"
    assert compatibility_series.particles_path_setter_value == "particles/"
    assert compatibility_series.attributes["meshesPath"] == "meshes/"
    assert compatibility_series.attributes["particlesPath"] == "particles/"


def test_record_metadata_and_store_fallbacks():
    """Test metadata and component storage fallbacks."""
    unit_dimension = SimpleNamespace(
        L="L",
        M="M",
        T="T",
        I="I",
        theta="theta",
        N="N",
        J="J",
    )
    record = FakeRecord("record")
    _set_record_metadata(
        record,
        SimpleNamespace(Unit_Dimension=unit_dimension),
        "charge",
    )
    assert record.unit_dimension == {"T": 1.0, "I": 1.0}

    fallback_record = StrictMetadataRecord()
    _set_record_metadata(
        fallback_record,
        SimpleNamespace(),
        "electric_field",
        macro_weighted=1,
        weighting_power=0.0,
    )
    assert fallback_record.attributes["unitDimension"].shape == (7,)
    assert fallback_record.attributes["timeOffset"] == 0.0
    assert fallback_record.attributes["macroWeighted"] == np.uint32(1)
    assert fallback_record.attributes["weightingPower"] == 0.0

    component = StrictRecordComponent("component")
    _store(component, np.array([1.0, 2.0]), SimpleNamespace(Dataset=FakeDataset))
    assert component.attributes["unitSI"] == 1.0


def test_write_meshes_skips_missing_records_and_uses_unit_fallbacks():
    """Test mesh writing branches for optional records and unit fallbacks."""
    iteration = SimpleNamespace(meshes=FakeMap(lambda name: StrictMesh(name)))
    output = {
        "electric_field": np.ones((1, 4, 3)),
        "dx": 0.25,
        "length": 1.0,
    }
    fake_io = SimpleNamespace(
        Dataset=FakeDataset,
        Geometry=SimpleNamespace(cartesian="cartesian"),
    )

    _write_meshes(iteration, output, 0, fake_io)

    assert set(iteration.meshes.entries) == {"E"}
    mesh = iteration.meshes.entries["E"]
    assert mesh.attributes["gridUnitSI"] == 1.0
    assert mesh.components["x"].data.shape == (4,)


@pytest.mark.parametrize("extension", OPENPMD_EXTENSIONS)
def test_write_openpmd_combined_series(monkeypatch, tmp_path, extension):
    """Test jaxincell._openpmd.write_openpmd with a fake openpmd_api module.

    Cases:
    - one combined series contains both mesh and particle records.
    - openpmd_iteration_stride selects only every Nth iteration.
    - root path attributes and sidecar output are written.
    """
    monkeypatch.setattr("jaxincell._openpmd.io", fake_openpmd_module())
    output = tiny_openpmd_output(
        tmp_path,
        openpmd_filename=str(tmp_path / f"jaxincell_output{extension}"),
        openpmd_iteration_stride=2,
        openpmd_meshes_path="custom_meshes/",
        openpmd_particles_path="custom_particles/",
    )

    paths = write_openpmd(output)

    assert paths["data"]["combined"] == str(tmp_path / f"jaxincell_output{extension}")
    assert (tmp_path / "jaxincell_output.pmd").read_text() == f"jaxincell_output{extension}\n"
    assert len(FakeSeries.created) == 1

    series = FakeSeries.created[0]
    assert series.path == str(tmp_path / f"jaxincell_output{extension}")
    assert series.attributes["meshesPath"] == "custom_meshes/"
    assert series.attributes["particlesPath"] == "custom_particles/"
    assert set(series.iterations.entries) == {0, 2}
    assert series.flushed is True
    assert series.closed is True

    iteration = series.iterations.entries[0]
    assert set(iteration.meshes.entries) >= {"E", "B", "J", "rho", "external_E", "external_B"}
    assert set(iteration.particles.entries) == {"electron_beam", "ions0"}
    assert iteration.meshes.entries["E"].components["x"].data.shape == (4,)
    assert iteration.particles.entries["electron_beam"].components["charge"].data.tolist() == [-1.0, -1.0]


@pytest.mark.parametrize("extension", OPENPMD_EXTENSIONS)
def test_write_openpmd_separate_series(monkeypatch, tmp_path, extension):
    """Test jaxincell._openpmd.write_openpmd separate particle and mesh output.

    Cases:
    - supported filename extensions are preserved.
    - mesh and particle records are written to separate series objects.
    - each generated series gets its own sidecar.
    """
    monkeypatch.setattr("jaxincell._openpmd.io", fake_openpmd_module())
    output = tiny_openpmd_output(
        tmp_path,
        openpmd_filename=str(tmp_path / f"split{extension}"),
        openpmd_separate_particles_and_meshes=True,
    )

    paths = write_openpmd(output)

    assert paths["data"]["meshes"] == str(tmp_path / f"split_meshes{extension}")
    assert paths["data"]["particles"] == str(tmp_path / f"split_particles{extension}")
    assert (tmp_path / "split_meshes.pmd").read_text() == f"split_meshes{extension}\n"
    assert (tmp_path / "split_particles.pmd").read_text() == f"split_particles{extension}\n"
    assert [series.path for series in FakeSeries.created] == [
        str(tmp_path / f"split_meshes{extension}"),
        str(tmp_path / f"split_particles{extension}"),
    ]
    assert FakeSeries.created[0].iterations.entries[0].meshes.entries
    assert not FakeSeries.created[0].iterations.entries[0].particles.entries
    assert FakeSeries.created[1].iterations.entries[0].particles.entries
    assert not FakeSeries.created[1].iterations.entries[0].meshes.entries


@pytest.mark.parametrize("extension", OPENPMD_EXTENSIONS)
def test_write_openpmd_file_based_series(monkeypatch, tmp_path, extension):
    """Test jaxincell._openpmd.write_openpmd fileBased output.

    Cases:
    - supported filename extensions are accepted.
    - fileBased output opens a templated series name.
    - sidecars list the templated data file basename.
    """
    monkeypatch.setattr("jaxincell._openpmd.io", fake_openpmd_module())
    output = tiny_openpmd_output(
        tmp_path,
        openpmd_filename=str(tmp_path / f"filebased{extension}"),
        openpmd_iteration_encoding="fileBased",
    )

    paths = write_openpmd(output)

    assert paths["data"]["combined"] == str(tmp_path / f"filebased_%06T{extension}")
    assert (tmp_path / "filebased.pmd").read_text() == f"filebased_%06T{extension}\n"
    assert len(FakeSeries.created) == 1

    series = FakeSeries.created[0]
    assert series.path == str(tmp_path / f"filebased_%06T{extension}")
    assert series.iteration_encoding == "fileBased"
    assert series.iteration_format == f"filebased_%06T{extension}"
    assert set(series.iterations.entries) == {0, 1, 2}


def test_write_openpmd_computed_time_and_particle_edge_cases(monkeypatch, tmp_path):
    """Test writer branches for computed time and particle naming edge cases."""
    monkeypatch.setattr("jaxincell._openpmd.io", fake_openpmd_module())
    output = tiny_openpmd_output(
        tmp_path,
        openpmd_filename=str(tmp_path / "particle_edges.h5"),
    )
    output.pop("time_array")
    output.pop("external_electric_field")
    output.pop("external_magnetic_field")
    output["solver_parameters"]["time_evolution_algorithm"] = 1
    output["species_parameters"] = {
        "electrons": {
            "_electrons0": {"user_label": "beam"},
            "_electrons1": {"user_label": "beam"},
        },
        "ions": {
            "_ions0": {"user_label": "beam"},
        },
    }

    write_openpmd(output)

    series = FakeSeries.created[0]
    assert series.iterations.entries[2].time == pytest.approx(0.2)
    iteration = series.iterations.entries[0]
    assert "external_E" not in iteration.meshes.entries
    assert "external_B" not in iteration.meshes.entries
    assert set(iteration.particles.entries) == {"beam", "beam_1"}
    assert (
        iteration.particles.entries["beam"].attributes["particlePush"]
        == "Implicit Crank-Nicolson"
    )


def test_simulation_run_writes_openpmd_when_enabled(monkeypatch, tmp_path):
    """Test automatic Simulation.run openPMD writing.

    Cases:
    - the default openpmd_output=False path does not call the writer.
    - enabling openpmd_output=True writes after output assembly.
    - the writer return value is added as output metadata.
    """
    called = []

    def fake_writer(output):
        called.append(output)
        return {"data": {"combined": output["export_parameters"]["openpmd_filename"]}, "sidecar": {}}

    parameters = base_simulation_parameters()
    monkeypatch.setattr("jaxincell._simulation.write_openpmd", fake_writer)
    default_output = Simulation(parameters).run()

    assert called == []
    assert "openpmd_files" not in default_output

    parameters = base_simulation_parameters()
    parameters["export_parameters"] = {
        **parameters.get("export_parameters", {}),
        "openpmd_output": True,
        "openpmd_filename": str(tmp_path / "auto.h5"),
    }
    output = Simulation(parameters).run()

    assert called
    assert output["openpmd_files"]["data"]["combined"] == str(tmp_path / "auto.h5")
