def test_build_domain_state_defaults_and_grid_geometry():
    """Test jaxincell._state_initialization.build_domain_state.

    Cases to implement:
    - length_y and length_z of zero fall back to length.
    - number_grid_points_y and number_grid_points_z of zero fall back to three.
    - dx, dt, grid endpoints, and box_size are hand-checked for a small domain.
    """


def test_initialize_species_phase_space_position_and_velocity_modes():
    """Test jaxincell._state_initialization.initialize_species_phase_space.

    Cases to implement:
    - deterministic positions use linspace on each axis when random_positions_* is false.
    - random_positions_* uses the supplied seeds and stays inside the box.
    - perturbation_amplitude/wavenumber and velocity_plus_minus_* modify the expected axes only.
    """


def test_make_particles_from_state_electron_reference_and_weight_logic():
    """Test jaxincell._state_initialization.make_particles_from_state.

    Cases to implement:
    - first electron species creates the electron_reference metadata.
    - ion creation before an electron reference raises the documented ValueError.
    - weight=0 triggers automatic weight calculation while nonzero weight is preserved.
    - seed_position_override replaces the derived local position seed.
    """


def test_initialize_particle_state_multi_species_lookups_and_speed_clipping():
    """Test jaxincell._state_initialization.initialize_particle_state.

    Cases to implement:
    - multiple electron and ion species concatenate positions, velocities, masses, charges, and weights.
    - species_index, unique_species_indices, integer_key_map, and lookup arrays are internally consistent.
    - velocities at or above the speed limit are clipped to 0.99 * speed_of_light.
    """


def test_initialize_field_state_default_and_provided_external_fields():
    """Test jaxincell._state_initialization.initialize_field_state.

    Cases to implement:
    - default external electric and magnetic fields are zero arrays of shape (G, 3).
    - provided external_field_parameters dictionaries with E/B arrays are converted to JAX arrays.
    - initial electric field x-component is produced from calculate_charge_density and E_from_Gauss_1D_Cartesian.
    """
