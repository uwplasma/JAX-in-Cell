from copy import deepcopy

import pytest

import jaxincell._routing as routing
from jaxincell import Simulation
from jaxincell._parameters._sections import PARAMETER_SECTIONS
from jaxincell._routing import (
    build_runtime_flat_parameter_routes,
    build_runtime_parameter_sections,
    build_runtime_species_label_routes,
    clean_runtime_input_parameters,
    iter_species_parameter_groups,
    merge_parameter_trees,
    put_parameter_path,
    route_flat_initial_parameters,
    route_nested_initial_species_parameters,
)
from tests.helpers import base_simulation_parameters, scalar


def test_merge_parameter_trees_recurses_without_mutating_inputs():
    """Test jaxincell._routing.merge_parameter_trees.

    Cases:
    - nested dictionaries merge recursively with override values taking precedence.
    - non-dictionary leaves replace entire branches.
    - base_parameters and override_parameters are not mutated.
    """
    base_parameters = {
        "domain_parameters": {"length": 1.0, "total_steps": 4},
        "species_parameters": {
            "ions": {
                "_ions0": {"drift_speed_x": 1.0, "mass_over_proton_mass": 1.0},
            },
        },
        "replace_me": {"nested": True},
    }
    override_parameters = {
        "domain_parameters": {"length": 2.0},
        "species_parameters": {
            "ions": {
                "_ions0": {"drift_speed_x": 3.0},
                "_ions1": {"drift_speed_x": 4.0},
            },
        },
        "replace_me": 7,
    }
    base_copy = deepcopy(base_parameters)
    override_copy = deepcopy(override_parameters)

    merged_parameters = merge_parameter_trees(base_parameters, override_parameters)

    assert merged_parameters["domain_parameters"] == {"length": 2.0, "total_steps": 4}
    assert merged_parameters["species_parameters"]["ions"]["_ions0"] == {
        "drift_speed_x": 3.0,
        "mass_over_proton_mass": 1.0,
    }
    assert merged_parameters["species_parameters"]["ions"]["_ions1"] == {"drift_speed_x": 4.0}
    assert merged_parameters["replace_me"] == 7
    assert base_parameters == base_copy
    assert override_parameters == override_copy


def test_merge_parameter_trees_result_is_independent_from_override_only_branches():
    """Test that override-only nested branches are copied into the merged result."""
    override_parameters = {"new_branch": {"nested": {"value": 1}}}

    merged_parameters = merge_parameter_trees({}, override_parameters)
    merged_parameters["new_branch"]["nested"]["value"] = 2

    assert override_parameters["new_branch"]["nested"]["value"] == 1


def test_put_parameter_path_creates_nested_containers():
    """Test jaxincell._routing.put_parameter_path.

    Cases:
    - a new multi-key path creates intermediate dictionaries.
    - an existing branch is reused.
    - writing a second value to the same path overwrites only that leaf.
    """
    container = {}

    put_parameter_path(
        container,
        ("species_parameters", "ions", "_ions0", "drift_speed_x"),
        1.0,
    )
    put_parameter_path(
        container,
        ("species_parameters", "ions", "_ions0", "mass_over_proton_mass"),
        2.0,
    )
    put_parameter_path(
        container,
        ("species_parameters", "ions", "_ions0", "drift_speed_x"),
        3.0,
    )
    put_parameter_path(container, ("length",), 4.0)

    assert container == {
        "species_parameters": {
            "ions": {
                "_ions0": {
                    "drift_speed_x": 3.0,
                    "mass_over_proton_mass": 2.0,
                },
            },
        },
        "length": 4.0,
    }


def test_iter_species_parameter_groups_loose_nested_and_strict_errors():
    """Test jaxincell._routing.iter_species_parameter_groups.

    Cases:
    - loose species dictionaries return a single unlabeled group.
    - nested species dictionaries return one group per label.
    - strict=False ignores non-dict species sections.
    - strict=True rejects non-dict species sections and mixed scalar nested values.
    """
    loose_values = {"drift_speed_x": 1.0, "mass_over_proton_mass": 2.0}
    nested_values = {
        "beam": {"drift_speed_x": 1.0},
        "bulk": {"drift_speed_x": 2.0},
    }

    assert list(iter_species_parameter_groups("ions", loose_values)) == [(None, loose_values)]
    assert list(iter_species_parameter_groups("ions", loose_values, strict=True)) == [(None, loose_values)]
    assert list(iter_species_parameter_groups("ions", {})) == [(None, {})]
    assert list(iter_species_parameter_groups("ions", {}, strict=True)) == [(None, {})]
    assert list(iter_species_parameter_groups("ions", nested_values)) == list(nested_values.items())
    assert list(iter_species_parameter_groups("ions", 3.0)) == []

    with pytest.raises(TypeError, match="Runtime input parameter 'ions' must be a dictionary"):
        list(iter_species_parameter_groups("ions", 3.0, strict=True))

    with pytest.raises(TypeError, match="Runtime input parameter 'ions\\.bulk' must be a dictionary"):
        list(iter_species_parameter_groups(
            "ions",
            {"beam": {"drift_speed_x": 1.0}, "bulk": 3.0},
            strict=True,
        ))


def test_route_nested_initial_species_parameters_splits_diff_cleaner_and_section_values():
    """Test jaxincell._routing.route_nested_initial_species_parameters.

    Cases:
    - differentiable species keys are written to differentiable_parameters and cleaner_input_parameters.
    - non-differentiable but valid species keys are routed into parameters['species_parameters'].
    - unknown species keys remain only in cleaner_input_parameters for later validation.
    """
    input_parameters = {
        "ions": {
            "beam": {
                "drift_speed_x": 3.0,
                "number_pseudoparticles": 5,
                "unknown_species_key": 11,
            },
            "tail": {
                "drift_speed_x": 4.0,
                "number_pseudoparticles": 6,
            },
        },
        "electrons": {
            "drift_speed_x": -2.0,
            "number_pseudoparticles": 7,
            "unknown_species_key": 13,
        },
    }
    input_parameters_copy = deepcopy(input_parameters)
    parameters = {
        "species_parameters": {
            "ions": {
                "existing": {
                    "number_pseudoparticles": 2,
                },
            },
        },
    }
    differentiable_parameters = {}
    cleaner_input_parameters = {}

    route_nested_initial_species_parameters(
        input_parameters,
        parameters,
        differentiable_parameters,
        cleaner_input_parameters,
    )

    assert scalar(differentiable_parameters["ions"]["beam"]["drift_speed_x"]) == 3.0
    assert scalar(differentiable_parameters["ions"]["tail"]["drift_speed_x"]) == 4.0
    assert scalar(differentiable_parameters["electrons"]["drift_speed_x"]) == -2.0
    assert parameters["species_parameters"]["ions"]["existing"]["number_pseudoparticles"] == 2
    assert parameters["species_parameters"]["ions"]["beam"]["number_pseudoparticles"] == 5
    assert parameters["species_parameters"]["ions"]["tail"]["number_pseudoparticles"] == 6
    assert parameters["species_parameters"]["electrons"]["number_pseudoparticles"] == 7
    assert cleaner_input_parameters["ions"]["beam"]["drift_speed_x"] == 3.0
    assert cleaner_input_parameters["ions"]["beam"]["number_pseudoparticles"] == 5
    assert cleaner_input_parameters["ions"]["beam"]["unknown_species_key"] == 11
    assert cleaner_input_parameters["electrons"]["unknown_species_key"] == 13
    assert "number_pseudoparticles" not in differentiable_parameters["ions"]["beam"]
    assert "number_pseudoparticles" not in differentiable_parameters["ions"]["tail"]
    assert "number_pseudoparticles" not in differentiable_parameters["electrons"]
    assert "unknown_species_key" not in parameters["species_parameters"]["ions"]["beam"]
    assert "unknown_species_key" not in parameters["species_parameters"]["electrons"]
    assert "unknown_species_key" not in differentiable_parameters["ions"]["beam"]
    assert "unknown_species_key" not in differentiable_parameters["electrons"]
    assert input_parameters == input_parameters_copy


def test_route_flat_initial_parameters_splits_diff_cleaner_section_and_unrouted_values():
    """Test jaxincell._routing.route_flat_initial_parameters.

    Cases:
    - flat differentiable keys are exposed and copied into cleaner_input_parameters.
    - flat non-differentiable keys route into their owning parameter section.
    - species sections are ignored because nested species routing handles them first.
    - unknown flat keys are copied into cleaner_input_parameters and returned as unrouted values.
    """
    input_parameters = {
        "length": 0.02,
        "filter_alpha": 0.25,
        "total_steps": 3,
        "filter_passes": 0,
        "ions": {
            "ions0": {
                "drift_speed_x": 5.0,
            },
        },
        "unknown_flat_parameter": 11,
    }
    input_parameters_copy = deepcopy(input_parameters)
    parameters = {
        "domain_parameters": {
            "number_grid_points": 4,
        },
    }
    differentiable_parameters = {}
    cleaner_input_parameters = {}

    unrouted_input_parameters = route_flat_initial_parameters(
        input_parameters,
        parameters,
        differentiable_parameters,
        cleaner_input_parameters,
    )

    assert scalar(differentiable_parameters["length"]) == 0.02
    assert scalar(differentiable_parameters["filter_alpha"]) == 0.25
    assert "total_steps" not in differentiable_parameters
    assert "filter_passes" not in differentiable_parameters
    assert "ions" not in differentiable_parameters
    assert "unknown_flat_parameter" not in differentiable_parameters

    assert cleaner_input_parameters == {
        "length": 0.02,
        "filter_alpha": 0.25,
        "unknown_flat_parameter": 11,
    }
    assert parameters["domain_parameters"] == {
        "number_grid_points": 4,
        "total_steps": 3,
    }
    assert parameters["solver_parameters"] == {
        "filter_passes": 0,
    }
    assert "species_parameters" not in parameters
    assert unrouted_input_parameters == {
        "unknown_flat_parameter": 11,
    }
    assert input_parameters == input_parameters_copy


def test_build_runtime_flat_parameter_routes_has_unique_section_routes():
    """Test jaxincell._routing.build_runtime_flat_parameter_routes.

    Cases:
    - each flat differentiable parameter maps to every section that owns it.
    - duplicate section routes are not inserted.
    - expected domain and solver differentiable keys are present.
    - nested species differentiable keys are intentionally not flat routes.
    """
    routes = build_runtime_flat_parameter_routes()

    assert routes["length"] == [("domain_parameters", "length")]
    assert routes["length_y"] == [("domain_parameters", "length_y")]
    assert routes["filter_alpha"] == [("solver_parameters", "filter_alpha")]
    assert "drift_speed_x" not in routes
    assert "mass_over_proton_mass" not in routes
    for key, parameter_routes in routes.items():
        assert len(parameter_routes) == len(set(parameter_routes)), key


def test_build_runtime_flat_parameter_routes_deduplicates_same_section_routes(monkeypatch):
    """Test duplicate suppression and cross-section routing for flat differentiable keys."""
    monkeypatch.setattr(
        routing,
        "PARAMETER_SECTIONS",
        {
            "section_a": {"differentiable": ["shared", "shared", "unique"]},
            "section_b": {"differentiable": ["shared"]},
        },
    )

    routes = routing.build_runtime_flat_parameter_routes()

    assert routes["shared"] == [
        ("section_a", "shared"),
        ("section_b", "shared"),
    ]
    assert routes["unique"] == [("section_a", "unique")]


def test_build_runtime_species_label_routes_includes_canonical_and_user_labels():
    """Test jaxincell._routing.build_runtime_species_label_routes.

    Cases:
    - canonical labels such as _ions0 route to themselves.
    - user labels route to the canonical label.
    - None user labels are skipped while canonical labels are still routed.
    - shared user labels route to every matching canonical species in stable order.
    - ion and electron label namespaces remain separate.
    - duplicate user/canonical labels do not duplicate routes.
    """
    species_parameters = {
        "electrons": {
            "_electrons0": {"user_label": "bulk"},
            "_electrons1": {"user_label": None},
            "_electrons2": {"user_label": "_electrons2"},
            "_electrons3": {},
        },
        "ions": {
            "_ions0": {"user_label": "bulk"},
            "_ions1": {"user_label": "_ions1"},
            "_ions2": {"user_label": "bulk"},
            "_ions3": {},
        },
    }

    routes = build_runtime_species_label_routes(species_parameters)

    assert set(routes) == {"electrons", "ions"}

    assert routes["electrons"]["_electrons0"] == ["_electrons0"]
    assert routes["electrons"]["bulk"] == ["_electrons0"]
    assert routes["electrons"]["_electrons1"] == ["_electrons1"]
    assert routes["electrons"]["_electrons2"] == ["_electrons2"]
    assert routes["electrons"]["_electrons3"] == ["_electrons3"]
    assert None not in routes["electrons"]

    assert routes["ions"]["_ions0"] == ["_ions0"]
    assert routes["ions"]["_ions1"] == ["_ions1"]
    assert routes["ions"]["_ions2"] == ["_ions2"]
    assert routes["ions"]["_ions3"] == ["_ions3"]
    assert routes["ions"]["bulk"] == ["_ions0", "_ions2"]
    assert None not in routes["ions"]

    for species_routes in routes.values():
        for label_routes in species_routes.values():
            assert len(label_routes) == len(set(label_routes))


def test_clean_runtime_input_parameters_validation_and_species_routing():
    """Test jaxincell._routing.clean_runtime_input_parameters.

    Cases:
    - None input produces empty parameter-section override dictionaries.
    - non-dict input raises TypeError.
    - invalid flat keys and non-differentiable species keys raise ValueError with useful paths.
    - valid flat keys with no available route raise the unrouted-parameter error.
    - loose species runtime values apply to all canonical species of that type.
    - unknown user species labels raise ValueError.
    """
    parameters = base_simulation_parameters()
    parameters["species_parameters"]["ions"]["beam"] = {
        "number_pseudoparticles": 2,
        "drift_speed_x": 0.0,
    }
    sim = Simulation(parameters)
    flat_routes = build_runtime_flat_parameter_routes()
    label_routes = build_runtime_species_label_routes(sim.species_parameters)

    cleaned_none = clean_runtime_input_parameters(
        None,
        flat_routes,
        label_routes,
        sim.species_parameters,
    )
    assert cleaned_none == {section_name: {} for section_name in PARAMETER_SECTIONS}

    with pytest.raises(TypeError, match="Runtime input_parameters must be a dictionary"):
        clean_runtime_input_parameters([], flat_routes, label_routes, sim.species_parameters)

    with pytest.raises(TypeError, match="Runtime input parameter 'ions' must be a dictionary"):
        clean_runtime_input_parameters(
            {"ions": 3.0},
            flat_routes,
            label_routes,
            sim.species_parameters,
        )

    with pytest.raises(TypeError, match="Runtime input parameter 'ions\\.beam' must be a dictionary"):
        clean_runtime_input_parameters(
            {"ions": {"beam": 3.0, "ions0": {"drift_speed_x": 1.0}}},
            flat_routes,
            label_routes,
            sim.species_parameters,
        )

    with pytest.raises(ValueError, match="Invalid parameter\\(s\\): unknown_parameter"):
        clean_runtime_input_parameters(
            {"unknown_parameter": 1.0},
            flat_routes,
            label_routes,
            sim.species_parameters,
        )

    with pytest.raises(ValueError, match="ions\\.ions0\\.number_pseudoparticles"):
        clean_runtime_input_parameters(
            {"ions": {"ions0": {"number_pseudoparticles": 3}}},
            flat_routes,
            label_routes,
            sim.species_parameters,
        )

    with pytest.raises(ValueError, match="ions\\.number_pseudoparticles"):
        clean_runtime_input_parameters(
            {"ions": {"number_pseudoparticles": 3}},
            flat_routes,
            label_routes,
            sim.species_parameters,
        )

    with pytest.raises(ValueError, match="Unrouted parameter\\(s\\): length"):
        clean_runtime_input_parameters(
            {"length": 2.0},
            {},
            label_routes,
            sim.species_parameters,
        )

    flat_cleaned = clean_runtime_input_parameters(
        {"length": 2.0, "filter_alpha": 0.25},
        flat_routes,
        label_routes,
        sim.species_parameters,
    )
    assert flat_cleaned["domain_parameters"] == {"length": 2.0}
    assert flat_cleaned["solver_parameters"] == {"filter_alpha": 0.25}
    assert flat_cleaned["species_parameters"] == {}

    loose_species_cleaned = clean_runtime_input_parameters(
        {"ions": {"drift_speed_x": 5.0}},
        flat_routes,
        label_routes,
        sim.species_parameters,
    )
    cleaned_ions = loose_species_cleaned["species_parameters"]["ions"]
    assert cleaned_ions["_ions0"]["drift_speed_x"] == 5.0
    assert cleaned_ions["_ions1"]["drift_speed_x"] == 5.0

    loose_electrons_cleaned = clean_runtime_input_parameters(
        {"electrons": {"drift_speed_x": -5.0}},
        flat_routes,
        label_routes,
        sim.species_parameters,
    )
    assert loose_electrons_cleaned["species_parameters"]["electrons"]["_electrons0"] == {
        "drift_speed_x": -5.0,
    }

    canonical_species_cleaned = clean_runtime_input_parameters(
        {"ions": {"_ions0": {"drift_speed_x": 2.0}}},
        flat_routes,
        label_routes,
        sim.species_parameters,
    )
    assert canonical_species_cleaned["species_parameters"]["ions"]["_ions0"] == {
        "drift_speed_x": 2.0,
    }

    labeled_species_cleaned = clean_runtime_input_parameters(
        {"ions": {"beam": {"mass_over_proton_mass": 3.0}}},
        flat_routes,
        label_routes,
        sim.species_parameters,
    )
    assert labeled_species_cleaned["species_parameters"]["ions"]["_ions1"] == {
        "mass_over_proton_mass": 3.0,
    }
    assert "_ions0" not in labeled_species_cleaned["species_parameters"]["ions"]

    with pytest.raises(ValueError, match="Could not find ions species 'missing'"):
        clean_runtime_input_parameters(
            {"ions": {"missing": {"drift_speed_x": 1.0}}},
            flat_routes,
            label_routes,
            sim.species_parameters,
        )


def test_build_runtime_parameter_sections_merges_all_sections():
    """Test jaxincell._routing.build_runtime_parameter_sections.

    Cases:
    - runtime overrides are recursively merged into every base parameter section.
    - absent runtime sections leave the base section unchanged.
    - nested species overrides preserve unrelated species parameters.
    - base parameter sections are not mutated.
    """
    base_parameter_sections = {
        "domain_parameters": {"length": 1.0, "total_steps": 2},
        "species_parameters": {
            "ions": {
                "_ions0": {
                    "drift_speed_x": 1.0,
                    "mass_over_proton_mass": 1.0,
                },
                "_ions1": {
                    "drift_speed_x": 2.0,
                    "mass_over_proton_mass": 4.0,
                },
            },
            "electrons": {
                "_electrons0": {
                    "drift_speed_x": -1.0,
                    "charge_over_elementary_charge": -1.0,
                },
            },
        },
        "external_field_parameters": {"external_electric_field": None},
        "source_parameters": {"source_term_active": 0},
        "solver_parameters": {"filter_alpha": 0.5, "filter_passes": 0},
        "export_parameters": {"openpmd_output": False},
    }
    input_parameters = {
        "domain_parameters": {"length": 3.0},
        "species_parameters": {
            "ions": {
                "_ions0": {"drift_speed_x": 7.0},
            },
        },
        "solver_parameters": {"filter_alpha": 0.25},
        "export_parameters": {"openpmd_output": True},
    }
    base_copy = deepcopy(base_parameter_sections)

    runtime_sections = build_runtime_parameter_sections(base_parameter_sections, input_parameters)

    assert runtime_sections["domain_parameters"] == {"length": 3.0, "total_steps": 2}
    assert runtime_sections["species_parameters"]["ions"]["_ions0"] == {
        "drift_speed_x": 7.0,
        "mass_over_proton_mass": 1.0,
    }
    assert runtime_sections["species_parameters"]["ions"]["_ions1"] == {
        "drift_speed_x": 2.0,
        "mass_over_proton_mass": 4.0,
    }
    assert runtime_sections["species_parameters"]["electrons"] == {
        "_electrons0": {
            "drift_speed_x": -1.0,
            "charge_over_elementary_charge": -1.0,
        },
    }
    assert runtime_sections["external_field_parameters"] == base_parameter_sections["external_field_parameters"]
    assert runtime_sections["source_parameters"] == base_parameter_sections["source_parameters"]
    assert runtime_sections["solver_parameters"] == {"filter_alpha": 0.25, "filter_passes": 0}
    assert runtime_sections["export_parameters"] == {"openpmd_output": True}
    assert base_parameter_sections == base_copy

    runtime_sections["source_parameters"]["source_term_active"] = 1
    assert base_parameter_sections["source_parameters"]["source_term_active"] == 0


def test_build_runtime_parameter_sections_result_is_independent_from_input_overrides():
    """Test that runtime section output can be mutated without mutating runtime overrides."""
    base_parameter_sections = {
        "domain_parameters": {},
    }
    input_parameters = {
        "domain_parameters": {
            "new_nested_branch": {
                "value": 1,
            },
        },
    }

    runtime_sections = build_runtime_parameter_sections(base_parameter_sections, input_parameters)
    runtime_sections["domain_parameters"]["new_nested_branch"]["value"] = 2

    assert input_parameters["domain_parameters"]["new_nested_branch"]["value"] == 1
