def test_merge_parameter_trees_recurses_without_mutating_inputs():
    """Test jaxincell._routing.merge_parameter_trees.

    Cases to implement:
    - nested dictionaries merge recursively with override values taking precedence.
    - non-dictionary leaves replace entire branches.
    - base_parameters and override_parameters are not mutated.
    """


def test_put_parameter_path_creates_nested_containers():
    """Test jaxincell._routing.put_parameter_path.

    Cases to implement:
    - a new multi-key path creates intermediate dictionaries.
    - an existing branch is reused.
    - writing a second value to the same path overwrites only that leaf.
    """


def test_iter_species_parameter_groups_loose_nested_and_strict_errors():
    """Test jaxincell._routing.iter_species_parameter_groups.

    Cases to implement:
    - loose species dictionaries return a single unlabeled group.
    - nested species dictionaries return one group per label.
    - strict=True rejects non-dict species sections and mixed scalar nested values.
    """


def test_route_nested_initial_species_parameters_splits_diff_cleaner_and_section_values():
    """Test jaxincell._routing.route_nested_initial_species_parameters.

    Cases to implement:
    - differentiable species keys are written to differentiable_parameters and cleaner_input_parameters.
    - non-differentiable but valid species keys are routed into parameters['species_parameters'].
    - unknown species keys remain only in cleaner_input_parameters for later validation.
    """


def test_build_runtime_flat_parameter_routes_has_unique_section_routes():
    """Test jaxincell._routing.build_runtime_flat_parameter_routes.

    Cases to implement:
    - each flat differentiable parameter maps to every section that owns it.
    - duplicate section routes are not inserted.
    - expected domain and solver differentiable keys are present.
    """


def test_build_runtime_species_label_routes_includes_canonical_and_user_labels():
    """Test jaxincell._routing.build_runtime_species_label_routes.

    Cases to implement:
    - canonical labels such as _ions0 route to themselves.
    - user labels such as ions0 route to the canonical label.
    - duplicate user/canonical labels do not duplicate routes.
    """


def test_clean_runtime_input_parameters_validation_and_species_routing():
    """Test jaxincell._routing.clean_runtime_input_parameters.

    Cases to implement:
    - None input produces empty parameter-section override dictionaries.
    - non-dict input raises TypeError.
    - invalid flat keys and non-differentiable species keys raise ValueError with useful paths.
    - loose species runtime values apply to all canonical species of that type.
    - unknown user species labels raise ValueError.
    """


def test_build_runtime_parameter_sections_merges_all_sections():
    """Test jaxincell._routing.build_runtime_parameter_sections.

    Cases to implement:
    - runtime overrides are recursively merged into every base parameter section.
    - absent runtime sections leave the base section unchanged.
    - nested species overrides preserve unrelated species parameters.
    """
