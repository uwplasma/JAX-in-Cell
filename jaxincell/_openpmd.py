import os
import re
from pathlib import Path

import numpy as np

from ._constants import speed_of_light
from ._parameters._species_definitions import SPECIES_TYPES

try:
    from .version import __version__
except Exception:
    __version__ = "unknown"

__all__ = [
    "openpmd_output_paths",
    "write_openpmd",
]

SUPPORTED_OPENPMD_EXTENSIONS = (".h5", ".bp")
DEFAULT_BASE_PATH = "/data/%T/"
UNIT_DIMENSION_ORDER = ("L", "M", "T", "I", "theta", "N", "J")
UNIT_DIMENSIONS = {
    "dimensionless": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "length": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "mass": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "charge": (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    "momentum": (1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0),
    "electric_field": (1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0),
    "magnetic_field": (0.0, 1.0, -2.0, -1.0, 0.0, 0.0, 0.0),
    "current_density": (-2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "charge_density": (-3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
}
MESH_RECORDS = (
    ("E", "electric_field", "vector", "electric_field", True),
    ("B", "magnetic_field", "vector", "magnetic_field", True),
    ("J", "current_density", "vector", "current_density", True),
    ("rho", "charge_density", "scalar", "charge_density", True),
    ("external_E", "external_electric_field", "vector", "electric_field", False),
    ("external_B", "external_magnetic_field", "vector", "magnetic_field", False),
)


def _import_openpmd_api():
    try:
        import openpmd_api as io
    except ImportError as exc:
        raise ImportError(
            "openPMD output requires the optional dependency 'openpmd-api'. "
            "Install it before setting solver_parameters['openpmd_output'] = True."
        ) from exc
    return io


def _set_attr(target, name, value):
    setter = getattr(target, "set_attribute", None)
    if setter is not None:
        setter(name, value)
    else:
        setattr(target, name, value)


def openpmd_output_paths(
    filename,
    overwrite=False,
    separate_particles_and_meshes=False,
    write_pmd_sidecar=True,
):
    filename = os.fspath(filename)
    root, extension = os.path.splitext(filename)
    if not extension:
        extension = ".h5"
    if extension.lower() not in SUPPORTED_OPENPMD_EXTENSIONS:
        supported = ", ".join(SUPPORTED_OPENPMD_EXTENSIONS)
        raise ValueError(
            f"openpmd_filename must end in one of {supported}. Got {filename!r}."
        )
    
    all_suffixes = ["_meshes", "_particles"] if separate_particles_and_meshes else [""]

    paths = {
        "data": [f"{root}{suffix}{extension}" for suffix in all_suffixes],
        "sidecar": [f"{root}{suffix}.pmd" for suffix in all_suffixes],
    }
    index = 0
    while not overwrite and any(
        os.path.exists(path)
        for path in (*paths["data"], *paths["sidecar"])
    ):
        paths = {
            "data": [f"{root}{index}{suffix}{extension}" for suffix in all_suffixes],
            "sidecar": [f"{root}{index}{suffix}.pmd" for suffix in all_suffixes],
        }
        index += 1

    if separate_particles_and_meshes:
        data_paths = {"meshes": paths["data"][0], "particles": paths["data"][1]}
        sidecar_paths = {"meshes": paths["sidecar"][0], "particles": paths["sidecar"][1]}
    else:
        data_paths = {"combined": paths["data"][0]}
        sidecar_paths = {"combined": paths["sidecar"][0]}
    
    if not write_pmd_sidecar:
        sidecar_paths = {}
    return {"data": data_paths, "sidecar": sidecar_paths}


def _open_series(path, io, meshes_path=None, particles_path=None):
    parent = Path(path).parent
    if str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)

    try:
        series = io.Series(path, io.Access.create)
    except Exception as exc:
        extension = os.path.splitext(path)[1].lower()
        raise RuntimeError(
            f"Could not create openPMD {extension} series at {path!r}. "
            "Check that the installed openpmd-api package supports this backend."
        ) from exc

    encoding = "groupBased"
    encoding_enum = getattr(io, "Iteration_Encoding", None)
    if encoding_enum is not None:
        encoding = (
            getattr(encoding_enum, "group_based", None)
            or getattr(encoding_enum, "groupBased", None)
            or encoding
        )
    if hasattr(series, "set_iteration_encoding"):
        try:
            series.set_iteration_encoding(encoding)
        except TypeError:
            series.set_iteration_encoding("groupBased")
    else:
        _set_attr(series, "iterationEncoding", "groupBased")
        _set_attr(series, "iterationFormat", DEFAULT_BASE_PATH)

    _set_attr(series, "software", "JAX-in-Cell")
    _set_attr(series, "softwareVersion", __version__)
    _set_attr(series, "basePath", DEFAULT_BASE_PATH)

    for path_value, attribute_name, setter_name, property_name in (
        (meshes_path, "meshesPath", "set_meshes_path", "meshes_path"),
        (particles_path, "particlesPath", "set_particles_path", "particles_path"),
    ):
        if path_value is None:
            continue
        normalized_path = path_value if path_value.endswith("/") else f"{path_value}/"
        setter = getattr(series, setter_name, None)
        if setter is not None:
            try:
                setter(normalized_path)
            except TypeError:
                pass
        try:
            setattr(series, property_name, normalized_path)
        except Exception:
            pass
        _set_attr(series, attribute_name, normalized_path)

    return series


def _set_record_metadata(
    record,
    io,
    dimension_name,
    *,
    macro_weighted=None,
    weighting_power=None,
):
    dimension = tuple(float(value) for value in UNIT_DIMENSIONS[dimension_name])
    unit_dimension = getattr(io, "Unit_Dimension", None)
    try:
        if unit_dimension is None:
            record.unit_dimension = dimension
        else:
            record.unit_dimension = {
                getattr(unit_dimension, key): value
                for key, value in zip(UNIT_DIMENSION_ORDER, dimension)
                if value != 0.0
            }
    except Exception:
        _set_attr(record, "unitDimension", np.asarray(dimension, dtype=np.float64))

    try:
        record.time_offset = 0.0
    except Exception:
        _set_attr(record, "timeOffset", 0.0)

    if macro_weighted is not None:
        _set_attr(record, "macroWeighted", np.uint32(macro_weighted))
    if weighting_power is not None:
        _set_attr(record, "weightingPower", float(weighting_power))


def _store(record_component, data, io):
    array = np.asarray(data, dtype=np.float64)
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.array(array, dtype=np.float64, copy=True, order="C")
    record_component.reset_dataset(io.Dataset(array.dtype, array.shape))
    record_component.store_chunk(array, [0] * array.ndim, array.shape)
    try:
        record_component.unit_SI = 1.0
    except Exception:
        _set_attr(record_component, "unitSI", 1.0)


def _write_meshes(iteration, output, iteration_index, io):
    for name, output_key, record_type, dimension_name, time_dependent in MESH_RECORDS:
        if output_key not in output:
            continue

        data = np.asarray(output[output_key])
        if time_dependent:
            data = data[iteration_index]

        mesh = iteration.meshes[name]
        mesh.geometry = io.Geometry.cartesian
        mesh.data_order = io.Data_Order.C if hasattr(io, "Data_Order") else "C"
        mesh.axis_labels = ["x"]
        mesh.grid_spacing = [float(np.asarray(output["dx"]))]
        mesh.grid_global_offset = [-0.5 * float(np.asarray(output["length"]))]
        try:
            mesh.grid_unit_SI = 1.0
        except Exception:
            pass
        try:
            mesh.unit_SI = 1.0
        except Exception:
            _set_attr(mesh, "gridUnitSI", 1.0)
        _set_record_metadata(mesh, io, dimension_name)

        if record_type == "vector":
            for component_index, component_name in enumerate(("x", "y", "z")):
                _store(mesh[component_name], data[:, component_index], io)
        else:
            scalar_component = getattr(io, "Mesh_Record_Component", None)
            scalar_key = "SCALAR" if scalar_component is None else scalar_component.SCALAR
            _store(mesh[scalar_key], data, io)


def _write_particles(iteration, output, iteration_index, io):
    species_names = []
    seen_names = {}
    for species_type in SPECIES_TYPES:
        for canonical_label, species in output["species_parameters"][species_type].items():
            species_name = re.sub(
                r"\W+",
                "_",
                str(species.get("user_label") or canonical_label),
            ).strip("_") or "species"
            if species_name in seen_names:
                seen_names[species_name] += 1
                species_name = f"{species_name}_{seen_names[species_name]}"
            else:
                seen_names[species_name] = 0
            species_names.append(species_name)

    species_index = np.asarray(output["species_integer_index"])
    all_positions = np.asarray(output["positions"], dtype=np.float64)
    all_velocities = np.asarray(output["velocities"], dtype=np.float64)
    all_weights = np.asarray(output["weights"], dtype=np.float64).reshape(-1)
    all_charges = np.asarray(output["charges"], dtype=np.float64).reshape(-1)
    all_masses = np.asarray(output["masses"], dtype=np.float64).reshape(-1)
    particle_push = (
        "Boris"
        if output["solver_parameters"]["time_evolution_algorithm"] == 0
        else "Implicit Crank-Nicolson"
    )

    for integer_index, species_name in enumerate(species_names):
        selection = species_index == integer_index
        positions = all_positions[iteration_index, selection, :]
        if positions.shape[0] == 0:
            continue

        velocities = all_velocities[iteration_index, selection, :]
        weights = all_weights[selection]
        macro_charges = all_charges[selection]
        macro_masses = all_masses[selection]
        charges = np.divide(
            macro_charges,
            weights,
            out=np.zeros_like(macro_charges),
            where=weights != 0,
        )
        masses = np.divide(
            macro_masses,
            weights,
            out=np.zeros_like(macro_masses),
            where=weights != 0,
        )
        speed_squared = np.sum(velocities**2, axis=1)
        gamma = 1.0 / np.sqrt(
            np.clip(1.0 - speed_squared / float(speed_of_light**2), 1e-15, None)
        )
        momentum = velocities * masses[:, None] * gamma[:, None]

        species_group = iteration.particles[species_name]
        for attribute_name, attribute_value in (
            ("particleShape", 1.0),
            ("currentDeposition", "other"),
            ("particlePush", particle_push),
            ("particleInterpolation", "other"),
            ("particleSmoothing", "none"),
        ):
            _set_attr(species_group, attribute_name, attribute_value)

        for record_name, values, dimension_name, macro_weighted, weighting_power in (
            ("position", positions, "length", 0, 0.0),
            ("positionOffset", np.zeros_like(positions), "length", 0, 0.0),
            ("momentum", momentum, "momentum", 0, 1.0),
        ):
            record = species_group[record_name]
            _set_record_metadata(
                record,
                io,
                dimension_name,
                macro_weighted=macro_weighted,
                weighting_power=weighting_power,
            )
            for component_index, component_name in enumerate(("x", "y", "z")):
                _store(record[component_name], values[:, component_index], io)

        for record_name, values, dimension_name, macro_weighted, weighting_power in (
            ("weighting", weights, "dimensionless", 1, 1.0),
            ("charge", charges, "charge", 0, 1.0),
            ("mass", masses, "mass", 0, 1.0),
        ):
            record = species_group[record_name]
            _set_record_metadata(
                record,
                io,
                dimension_name,
                macro_weighted=macro_weighted,
                weighting_power=weighting_power,
            )
            _store(record, values, io)


def _write_iterations(series, output, io, iteration_stride, write_meshes, write_particles):
    total_steps = int(output["total_steps"])
    for iteration_index in range(0, total_steps, iteration_stride):
        iteration = series.iterations[int(iteration_index)]
        if "time_array" in output:
            iteration.time = float(np.asarray(output["time_array"])[iteration_index])
        else:
            iteration.time = iteration_index * float(np.asarray(output["dt"]))
        iteration.dt = float(np.asarray(output["dt"]))
        iteration.time_unit_SI = 1.0

        if write_meshes:
            _write_meshes(iteration, output, iteration_index, io)
        if write_particles:
            _write_particles(iteration, output, iteration_index, io)

    series.flush()
    series.close()


def write_openpmd(output, solver_parameters=None):
    if solver_parameters is None:
        solver_parameters = output["solver_parameters"]
    if solver_parameters["openpmd_iteration_encoding"] != "groupBased":
        raise NotImplementedError("Only groupBased openPMD output is implemented.")

    paths = openpmd_output_paths(
        solver_parameters["openpmd_filename"],
        overwrite=solver_parameters["openpmd_overwrite"],
        separate_particles_and_meshes=solver_parameters[
            "openpmd_separate_particles_and_meshes"
        ],
        write_pmd_sidecar=solver_parameters["openpmd_write_pmd_sidecar"],
    )
    io = _import_openpmd_api()

    if solver_parameters["openpmd_separate_particles_and_meshes"]:
        mesh_series = _open_series(
            paths["data"]["meshes"],
            io,
            meshes_path=solver_parameters["openpmd_meshes_path"],
        )
        _write_iterations(
            mesh_series,
            output,
            io,
            solver_parameters["openpmd_iteration_stride"],
            write_meshes=True,
            write_particles=False,
        )

        particle_series = _open_series(
            paths["data"]["particles"],
            io,
            particles_path=solver_parameters["openpmd_particles_path"],
        )
        _write_iterations(
            particle_series,
            output,
            io,
            solver_parameters["openpmd_iteration_stride"],
            write_meshes=False,
            write_particles=True,
        )
    else:
        series = _open_series(
            paths["data"]["combined"],
            io,
            meshes_path=solver_parameters["openpmd_meshes_path"],
            particles_path=solver_parameters["openpmd_particles_path"],
        )
        _write_iterations(
            series,
            output,
            io,
            solver_parameters["openpmd_iteration_stride"],
            write_meshes=True,
            write_particles=True,
        )

    for key, sidecar_path in paths["sidecar"].items():
        parent = Path(sidecar_path).parent
        if str(parent) != ".":
            parent.mkdir(parents=True, exist_ok=True)
        with open(sidecar_path, "w", encoding="utf-8") as sidecar:
            sidecar.write(f"{os.path.basename(paths['data'][key])}\n")

    return paths
