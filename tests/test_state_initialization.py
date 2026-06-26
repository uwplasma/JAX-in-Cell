import numpy as np
import pytest
import jax.numpy as jnp
from jax.random import PRNGKey, normal, uniform

from jaxincell._constants import (
    elementary_charge,
    epsilon_0,
    mass_electron,
    mass_proton,
    speed_of_light,
)
from jaxincell._fields import E_from_Gauss_1D_Cartesian
from jaxincell._parameters._domain_parameters import clean_and_initialize_domain_parameters
from jaxincell._parameters._external_field_parameters import clean_and_initialize_external_field_parameters
from jaxincell._parameters._solver_parameters import clean_and_initialize_solver_parameters
from jaxincell._parameters._species_definitions import (
    DEFAULT_ELECTRON_PARAMETERS,
    DEFAULT_ION_PARAMETERS,
)
from jaxincell._sources import calculate_charge_density
from jaxincell._state_initialization import (
    build_domain_state,
    initialize_field_state,
    initialize_particle_state,
    initialize_species_phase_space,
    make_particles_from_state,
)
from tests.helpers import scalar


def domain_parameters(**overrides):
    parameters = {
        "total_steps": 2,
        "timestep_over_spatialstep_times_c": 0.5,
        "number_grid_points": 4,
        "number_grid_points_y": 0,
        "number_grid_points_z": 0,
        "length": 4.0,
        "length_y": 0.0,
        "length_z": 0.0,
        "particle_BC_left": 0,
        "particle_BC_right": 0,
        "field_BC_left": 0,
        "field_BC_right": 0,
    }
    parameters.update(overrides)
    return clean_and_initialize_domain_parameters(parameters)


def solver_parameters(**overrides):
    parameters = {
        "print_info": False,
        "field_solver": 0,
        "relativistic": False,
        "time_evolution_algorithm": 0,
        "max_number_of_Picard_iterations_implicit_CN": 2,
        "number_of_particle_substeps_implicit_CN": 1,
        "tolerance_Picard_iterations_implicit_CN": 1e-6,
        "filter_passes": 0,
        "filter_alpha": 0.5,
        "filter_strides": (1,),
        "seed": 1701,
    }
    parameters.update(overrides)
    return clean_and_initialize_solver_parameters(parameters)


def electron_species(**overrides):
    values = {**DEFAULT_ELECTRON_PARAMETERS}
    values.update(overrides)
    return values


def ion_species(**overrides):
    values = {**DEFAULT_ION_PARAMETERS}
    values.update(overrides)
    return values


def expected_random_axis(seed, number_particles, length):
    return uniform(
        PRNGKey(seed),
        shape=(number_particles,),
        minval=-length / 2,
        maxval=length / 2,
    )


def expected_velocity_axis(seed, species, axis, number_particles):
    return (
        species[f"vth_over_c_{axis}"]
        * speed_of_light
        / jnp.sqrt(2)
        * normal(PRNGKey(seed), shape=(number_particles,))
        + species[f"drift_speed_{axis}"]
    )


def test_build_domain_state_defaults_and_grid_geometry():
    """Test jaxincell._state_initialization.build_domain_state.

    Cases covered:
    - length_y and length_z of zero fall back to length.
    - dx, dt, grid endpoints, and box_size are hand-checked for a small domain.
    """
    parameters = domain_parameters()
    state = build_domain_state(parameters)

    assert tuple(scalar(value) for value in state["box_size"]) == (4.0, 4.0, 4.0)
    assert scalar(state["dx"]) == pytest.approx(1.0)
    assert scalar(state["dt"]) == pytest.approx(0.5 / speed_of_light)
    np.testing.assert_allclose(
        np.asarray(state["grid"]),
        np.array([-1.5, -0.5, 0.5, 1.5]),
        rtol=0,
        atol=1e-12,
    )

    transverse_parameters = domain_parameters(
        length_y=6.0,
        length_z=8.0,
        number_grid_points_y=5,
        number_grid_points_z=7,
    )
    transverse_state = build_domain_state(transverse_parameters)
    assert tuple(scalar(value) for value in transverse_state["box_size"]) == (4.0, 6.0, 8.0)

    mixed_transverse_state = build_domain_state(domain_parameters(length_y=0.0, length_z=8.0))
    assert tuple(scalar(value) for value in mixed_transverse_state["box_size"]) == (4.0, 4.0, 8.0)


def test_initialize_species_phase_space_position_and_velocity_modes():
    """Test jaxincell._state_initialization.initialize_species_phase_space.

    Cases covered:
    - deterministic positions use linspace on each axis when random_positions_* is false.
    - random_positions_* uses the supplied seeds and stays inside the box.
    - perturbation_amplitude/wavenumber and velocity_plus_minus_* modify the expected axes only.
    """
    number_particles = 4
    box_size = (2.0, 4.0, 6.0)
    unflipped_species = electron_species(
        random_positions_y=True,
        perturbation_amplitude_z=0.1,
        perturbation_wavenumber_z=1.0,
        random_positions_z=False,
        vth_over_c_x=0.01,
        drift_speed_x=2.0,
        drift_speed_y=3.0,
        drift_speed_z=4.0,
    )
    flipped_species = {
        **unflipped_species,
        "velocity_plus_minus_x": True,
        "velocity_plus_minus_y": False,
        "velocity_plus_minus_z": False,
    }

    unflipped_positions, unflipped_velocities = initialize_species_phase_space(
        unflipped_species,
        seed_position=11,
        seed_velocity=17,
        number_particles=number_particles,
        box_size=box_size,
    )

    positions, velocities = initialize_species_phase_space(
        flipped_species,
        seed_position=11,
        seed_velocity=17,
        number_particles=number_particles,
        box_size=box_size,
    )
    repeated_positions, repeated_velocities = initialize_species_phase_space(
        flipped_species,
        seed_position=11,
        seed_velocity=17,
        number_particles=number_particles,
        box_size=box_size,
    )

    expected_x = jnp.linspace(-1.0, 1.0, number_particles)
    base_z = jnp.linspace(-3.0, 3.0, number_particles)
    expected_z = base_z + 0.1 * jnp.sin((2 * jnp.pi / 6.0) * base_z)

    assert jnp.allclose(positions[:, 0], expected_x)
    assert jnp.allclose(positions, unflipped_positions)
    assert jnp.all(positions[:, 1] >= -2.0)
    assert jnp.all(positions[:, 1] <= 2.0)
    assert jnp.allclose(positions[:, 2], expected_z)
    assert jnp.allclose(positions, repeated_positions)
    assert jnp.allclose(velocities, repeated_velocities)
    assert jnp.allclose(
        velocities[:, 0],
        unflipped_velocities[:, 0] * (-1) ** jnp.arange(number_particles),
    )
    assert jnp.allclose(velocities[:, 1:], unflipped_velocities[:, 1:])


def test_make_particles_from_state_electron_reference_and_weight_logic():
    """Test jaxincell._state_initialization.make_particles_from_state.

    Cases covered:
    - first electron species creates the electron_reference metadata.
    - ion creation before an electron reference raises the documented ValueError.
    - ion creation succeeds once an electron reference exists.
    - unknown species types raise ValueError.
    - weight=0 triggers automatic weight calculation while nonzero weight is preserved.
    - seed_position_override replaces the derived local position seed.
    """
    domain = domain_parameters(length=2.0, number_grid_points=4)
    domain_state = build_domain_state(domain)
    solver = solver_parameters(seed=10)
    electron = electron_species(
        number_pseudoparticles=3,
        weight=2.0,
        charge_over_elementary_charge=-1.0,
        vth_over_c_x=0.01,
        vth_over_c_y=0.02,
        vth_over_c_z=0.03,
    )

    electron_state, electron_reference = make_particles_from_state(
        electron,
        "electrons",
        rng_index=0,
        domain_parameters=domain,
        solver_parameters=solver,
        domain_state=domain_state,
        electron_reference=None,
    )

    assert scalar(electron_state["charge"]) == pytest.approx(-elementary_charge)
    assert scalar(electron_state["mass"]) == pytest.approx(mass_electron)
    assert scalar(electron_state["charge_mass"]) == pytest.approx(-elementary_charge / mass_electron)
    assert jnp.allclose(electron_state["weights"], 2.0)
    assert scalar(electron_reference["vth_electrons_over_c"]) == pytest.approx(0.03)
    assert scalar(electron_reference["vth_electrons"]) == pytest.approx(0.03 * speed_of_light)
    assert scalar(electron_reference["charge_electrons"]) == pytest.approx(-elementary_charge)

    automatic_weight_electron = electron_species(
        number_pseudoparticles=3,
        grid_points_per_Debye_length=1.0,
        weight=0.0,
        charge_over_elementary_charge=-1.0,
        vth_over_c_x=0.01,
    )
    automatic_state, _ = make_particles_from_state(
        automatic_weight_electron,
        "electrons",
        rng_index=0,
        domain_parameters=domain,
        solver_parameters=solver,
        domain_state=domain_state,
        electron_reference=None,
    )
    assert jnp.all(jnp.isfinite(automatic_state["weights"]))
    assert jnp.all(automatic_state["weights"] > 0)
    expected_auto_weight = (
        epsilon_0
        * mass_electron
        * speed_of_light**2
        / elementary_charge**2
        * domain["number_grid_points"]**2
        / domain_state["box_size"][0]
        / (2 * automatic_weight_electron["number_pseudoparticles"])
        * 0.01**2
        / (1 / automatic_weight_electron["grid_points_per_Debye_length"])**2
    )
    assert jnp.allclose(automatic_state["weights"], expected_auto_weight)

    ion = ion_species(
        number_pseudoparticles=3,
        charge_over_elementary_charge=1.0,
        mass_over_proton_mass=2.0,
    )
    with pytest.raises(ValueError, match="Electron reference species must be initialized before ions"):
        make_particles_from_state(
            ion,
            "ions",
            rng_index=0,
            domain_parameters=domain,
            solver_parameters=solver,
            domain_state=domain_state,
            electron_reference=None,
        )

    ion_success = ion_species(
        number_pseudoparticles=3,
        weight=2.0,
        charge_over_elementary_charge=1.0,
        mass_over_proton_mass=2.0,
    )
    ion_state, _ = make_particles_from_state(
        ion_success,
        "ions",
        rng_index=0,
        domain_parameters=domain,
        solver_parameters=solver,
        domain_state=domain_state,
        electron_reference=electron_reference,
    )
    assert scalar(ion_state["charge"]) == pytest.approx(elementary_charge)
    assert scalar(ion_state["mass"]) == pytest.approx(2.0 * mass_proton)
    assert scalar(ion_state["charge_mass"]) == pytest.approx(elementary_charge / (2.0 * mass_proton))
    assert jnp.allclose(ion_state["weights"], 2.0)

    with pytest.raises(ValueError, match="Unknown species type"):
        make_particles_from_state(
            electron,
            "dust",
            rng_index=0,
            domain_parameters=domain,
            solver_parameters=solver,
            domain_state=domain_state,
            electron_reference=None,
        )

    seeded_species = electron_species(
        number_pseudoparticles=3,
        random_positions_x=True,
        random_positions_y=True,
        random_positions_z=True,
        seed_position_override=True,
        seed_position=123,
        vth_over_c_x=0.01,
    )
    first_seed_state, _ = make_particles_from_state(
        seeded_species,
        "electrons",
        rng_index=0,
        domain_parameters=domain,
        solver_parameters=solver_parameters(seed=10),
        domain_state=domain_state,
        electron_reference=None,
    )
    second_seed_state, _ = make_particles_from_state(
        seeded_species,
        "electrons",
        rng_index=0,
        domain_parameters=domain,
        solver_parameters=solver_parameters(seed=999),
        domain_state=domain_state,
        electron_reference=None,
    )
    expected_override_positions = jnp.stack(
        [
            expected_random_axis(124, 3, 2.0),
            expected_random_axis(125, 3, 2.0),
            expected_random_axis(126, 3, 2.0),
        ],
        axis=1,
    )
    assert jnp.allclose(first_seed_state["positions"], second_seed_state["positions"])
    assert jnp.allclose(first_seed_state["positions"], expected_override_positions)
    assert not jnp.allclose(first_seed_state["velocities"][:, 0], second_seed_state["velocities"][:, 0])


def test_initial_phase_space_overrides_replace_generated_species_block():
    """Test per-species initial_positions and initial_velocities overrides.

    Cases covered:
    - overrides replace generated phase space for the selected species.
    - another species can continue using generated phase space.
    - raw list overrides are coerced to JAX arrays before concatenation.
    """
    number_particles = 3
    domain = domain_parameters(length=3.0, length_y=3.0, length_z=3.0)
    solver = solver_parameters(seed=21)
    domain_state = build_domain_state(domain)
    electron_positions = [
        [-0.5, -0.4, -0.3],
        [0.0, 0.1, 0.2],
        [0.5, 0.6, 0.7],
    ]
    electron_velocities = [
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8],
    ]
    species_parameters = {
        "electrons": {
            "_electrons0": electron_species(
                number_pseudoparticles=number_particles,
                weight=1.0,
                initial_positions=electron_positions,
                initial_velocities=electron_velocities,
            ),
        },
        "ions": {
            "_ions0": ion_species(
                number_pseudoparticles=number_particles,
                weight=1.0,
            ),
        },
    }

    particle_state = initialize_particle_state(
        species_parameters,
        domain,
        solver,
        domain_state,
    )

    electron_slice = slice(0, number_particles)
    ion_slice = slice(number_particles, 2 * number_particles)
    generated_ion_positions, generated_ion_velocities = initialize_species_phase_space(
        species_parameters["ions"]["_ions0"],
        seed_position=solver["seed"],
        seed_velocity=solver["seed"] + 6,
        number_particles=number_particles,
        box_size=domain_state["box_size"],
    )

    assert jnp.allclose(particle_state["positions"][electron_slice], jnp.asarray(electron_positions))
    assert jnp.allclose(particle_state["velocities"][electron_slice], jnp.asarray(electron_velocities))
    assert jnp.allclose(particle_state["positions"][ion_slice], generated_ion_positions)
    assert jnp.allclose(particle_state["velocities"][ion_slice], generated_ion_velocities)


def test_initialize_particle_state_uses_main_branch_seed_offsets_for_base_species():
    """Test base electron and ion RNG seed compatibility with the main branch.

    Cases covered:
    - first electron and first ion species share the historical position seeds.
    - first electron velocities use the historical electron velocity seeds.
    - first ion velocities use the historical ion velocity seeds.
    - electron and ion velocity streams are not accidentally identical.
    """
    seed = 50
    number_particles = 3
    domain = domain_parameters(length=2.0, length_y=2.0, length_z=2.0)
    solver = solver_parameters(seed=seed)
    domain_state = build_domain_state(domain)
    random_overrides = {
        "number_pseudoparticles": number_particles,
        "random_positions_x": True,
        "random_positions_y": True,
        "random_positions_z": True,
        "vth_over_c_x": 0.001,
        "vth_over_c_y": 0.002,
        "vth_over_c_z": 0.003,
        "perturbation_amplitude_x": 0.0,
        "perturbation_amplitude_y": 0.0,
        "perturbation_amplitude_z": 0.0,
    }
    random_electron = electron_species(**random_overrides)
    random_ion = ion_species(**random_overrides)
    species_parameters = {
        "electrons": {
            "_electrons0": random_electron,
        },
        "ions": {
            "_ions0": random_ion,
        },
    }

    particle_state = initialize_particle_state(
        species_parameters,
        domain,
        solver,
        domain_state,
    )

    electron_slice = slice(0, number_particles)
    ion_slice = slice(number_particles, 2 * number_particles)
    expected_positions = jnp.stack(
        [
            expected_random_axis(seed + 1, number_particles, 2.0),
            expected_random_axis(seed + 2, number_particles, 2.0),
            expected_random_axis(seed + 3, number_particles, 2.0),
        ],
        axis=1,
    )
    expected_electron_velocities = jnp.stack(
        [
            expected_velocity_axis(seed + 7, random_electron, "x", number_particles),
            expected_velocity_axis(seed + 8, random_electron, "y", number_particles),
            expected_velocity_axis(seed + 9, random_electron, "z", number_particles),
        ],
        axis=1,
    )
    expected_ion_velocities = jnp.stack(
        [
            expected_velocity_axis(seed + 10, random_ion, "x", number_particles),
            expected_velocity_axis(seed + 11, random_ion, "y", number_particles),
            expected_velocity_axis(seed + 12, random_ion, "z", number_particles),
        ],
        axis=1,
    )

    assert jnp.allclose(particle_state["positions"][electron_slice], expected_positions)
    assert jnp.allclose(particle_state["positions"][ion_slice], expected_positions)
    assert jnp.allclose(particle_state["velocities"][electron_slice], expected_electron_velocities)
    assert jnp.allclose(particle_state["velocities"][ion_slice], expected_ion_velocities)
    assert not jnp.allclose(
        particle_state["velocities"][electron_slice],
        particle_state["velocities"][ion_slice],
    )


def test_initialize_particle_state_preserves_extra_species_seed_schedule():
    """Test non-primary species RNG seed compatibility with legacy extra species.

    Cases covered:
    - the first non-primary species uses the old seed + 12 extra-species schedule.
    - later non-primary species advance by the old six-seed spacing.
    - position and velocity seeds for extra species preserve the old paired local seed behavior.
    """
    seed = 50
    number_particles = 3
    domain = domain_parameters(length=2.0, length_y=2.0, length_z=2.0)
    solver = solver_parameters(seed=seed)
    domain_state = build_domain_state(domain)
    random_overrides = {
        "number_pseudoparticles": number_particles,
        "random_positions_x": True,
        "random_positions_y": True,
        "random_positions_z": True,
        "vth_over_c_x": 0.001,
        "vth_over_c_y": 0.002,
        "vth_over_c_z": 0.003,
        "perturbation_amplitude_x": 0.0,
        "perturbation_amplitude_y": 0.0,
        "perturbation_amplitude_z": 0.0,
    }
    random_electron = electron_species(**random_overrides)
    random_ion = ion_species(**random_overrides)
    species_parameters = {
        "electrons": {
            "_electrons0": random_electron,
            "_beam": random_electron,
        },
        "ions": {
            "_ions0": random_ion,
            "_beam_neutralizer": random_ion,
        },
    }

    particle_state = initialize_particle_state(
        species_parameters,
        domain,
        solver,
        domain_state,
    )

    first_extra_slice = slice(number_particles, 2 * number_particles)
    second_extra_slice = slice(3 * number_particles, 4 * number_particles)
    expected_first_extra_positions, expected_first_extra_velocities = initialize_species_phase_space(
        random_electron,
        seed_position=seed + 12,
        seed_velocity=seed + 12,
        number_particles=number_particles,
        box_size=domain_state["box_size"],
    )
    expected_second_extra_positions, expected_second_extra_velocities = initialize_species_phase_space(
        random_ion,
        seed_position=seed + 18,
        seed_velocity=seed + 18,
        number_particles=number_particles,
        box_size=domain_state["box_size"],
    )

    assert jnp.allclose(particle_state["positions"][first_extra_slice], expected_first_extra_positions)
    assert jnp.allclose(particle_state["velocities"][first_extra_slice], expected_first_extra_velocities)
    assert jnp.allclose(particle_state["positions"][second_extra_slice], expected_second_extra_positions)
    assert jnp.allclose(particle_state["velocities"][second_extra_slice], expected_second_extra_velocities)


def test_initialize_particle_state_multi_species_lookups_and_speed_clipping():
    """Test jaxincell._state_initialization.initialize_particle_state.

    Cases covered:
    - multiple electron and ion species concatenate positions, velocities, masses, charges, and weights.
    - species_index, unique_species_indices, integer_key_map, and lookup arrays are internally consistent.
    - velocities at or above the speed limit are clipped to 0.99 * speed_of_light.
    """
    domain = domain_parameters(length=4.0, number_grid_points=4)
    solver = solver_parameters(seed=42)
    domain_state = build_domain_state(domain)
    species_parameters = {
        "electrons": {
            "_electrons0": electron_species(
                number_pseudoparticles=2,
                weight=2.0,
                charge_over_elementary_charge=-1.0,
            ),
            "_electrons1": electron_species(
                number_pseudoparticles=1,
                weight=3.0,
                charge_over_elementary_charge=-2.0,
                drift_speed_x=2.0 * speed_of_light,
                drift_speed_y=-2.0 * speed_of_light,
            ),
        },
        "ions": {
            "_ions0": ion_species(
                number_pseudoparticles=2,
                weight=4.0,
                charge_over_elementary_charge=1.0,
                mass_over_proton_mass=2.0,
            ),
            "_ions1": ion_species(
                number_pseudoparticles=1,
                weight=5.0,
                charge_over_elementary_charge=2.0,
                mass_over_proton_mass=3.0,
            ),
        },
    }

    particle_state = initialize_particle_state(
        species_parameters,
        domain,
        solver,
        domain_state,
    )

    expected_unique_species = [
        "electrons._electrons0",
        "electrons._electrons1",
        "ions._ions0",
        "ions._ions1",
    ]
    assert particle_state["positions"].shape == (6, 3)
    assert particle_state["velocities"].shape == (6, 3)
    assert particle_state["weights"].shape == (6, 1)
    assert particle_state["charges"].shape == (6, 1)
    assert particle_state["masses"].shape == (6, 1)
    assert particle_state["charge_to_mass_ratios"].shape == (6, 1)
    assert particle_state["unique_species_indices"] == expected_unique_species
    assert particle_state["species_index"] == [
        "electrons._electrons0",
        "electrons._electrons0",
        "electrons._electrons1",
        "ions._ions0",
        "ions._ions0",
        "ions._ions1",
    ]
    assert particle_state["integer_key_map"] == {
        species_identifier: index
        for index, species_identifier in enumerate(expected_unique_species)
    }

    expected_charges = np.array([
        -elementary_charge * 2.0,
        -elementary_charge * 2.0,
        -2.0 * elementary_charge * 3.0,
        elementary_charge * 4.0,
        elementary_charge * 4.0,
        2.0 * elementary_charge * 5.0,
    ]).reshape((-1, 1))
    expected_masses = np.array([
        mass_electron * 2.0,
        mass_electron * 2.0,
        mass_electron * 3.0,
        2.0 * mass_proton * 4.0,
        2.0 * mass_proton * 4.0,
        3.0 * mass_proton * 5.0,
    ]).reshape((-1, 1))
    expected_charge_to_mass_ratios = np.array([
        -elementary_charge / mass_electron,
        -elementary_charge / mass_electron,
        -2.0 * elementary_charge / mass_electron,
        elementary_charge / (2.0 * mass_proton),
        elementary_charge / (2.0 * mass_proton),
        2.0 * elementary_charge / (3.0 * mass_proton),
    ]).reshape((-1, 1))
    expected_weights = np.array([2.0, 2.0, 3.0, 4.0, 4.0, 5.0]).reshape((-1, 1))
    expected_charge_integer_lookup = np.array([
        -elementary_charge,
        -2.0 * elementary_charge,
        elementary_charge,
        2.0 * elementary_charge,
    ])
    expected_mass_integer_lookup = np.array([
        mass_electron,
        mass_electron,
        2.0 * mass_proton,
        3.0 * mass_proton,
    ])
    expected_charge_mass_integer_lookup = np.array([
        -elementary_charge / mass_electron,
        -2.0 * elementary_charge / mass_electron,
        elementary_charge / (2.0 * mass_proton),
        2.0 * elementary_charge / (3.0 * mass_proton),
    ])

    np.testing.assert_allclose(
        np.asarray(particle_state["weights"]),
        expected_weights,
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(np.asarray(particle_state["charges"]), expected_charges, rtol=1e-6, atol=0)
    np.testing.assert_allclose(np.asarray(particle_state["masses"]), expected_masses, rtol=1e-6, atol=0)
    np.testing.assert_allclose(
        np.asarray(particle_state["charge_to_mass_ratios"]),
        expected_charge_to_mass_ratios,
        rtol=1e-6,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(particle_state["charge_integer_lookup"]),
        expected_charge_integer_lookup,
        rtol=1e-6,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(particle_state["mass_integer_lookup"]),
        expected_mass_integer_lookup,
        rtol=1e-6,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(particle_state["charge_mass_integer_lookup"]),
        expected_charge_mass_integer_lookup,
        rtol=1e-6,
        atol=0,
    )
    assert scalar(particle_state["charge_lookup"]["ions._ions1"]) == pytest.approx(2.0 * elementary_charge)
    assert scalar(particle_state["mass_lookup"]["ions._ions1"]) == pytest.approx(3.0 * mass_proton)
    assert scalar(particle_state["charge_mass_lookup"]["ions._ions1"]) == pytest.approx(
        2.0 * elementary_charge / (3.0 * mass_proton)
    )
    assert scalar(particle_state["velocities"][2, 0]) == pytest.approx(0.99 * speed_of_light)
    assert scalar(particle_state["velocities"][2, 1]) == pytest.approx(-0.99 * speed_of_light)
    assert scalar(particle_state["vth_electrons"]) == pytest.approx(0.0)
    assert scalar(particle_state["vth_electrons_over_c"]) == pytest.approx(0.0)
    assert scalar(particle_state["charge_electrons"]) == pytest.approx(-elementary_charge)


def test_initialize_field_state_default_and_provided_external_fields():
    """Test jaxincell._state_initialization.initialize_field_state.

    Cases covered:
    - default external electric and magnetic fields are zero arrays of shape (G, 3).
    - provided external_field_parameters dictionaries with E/B arrays are converted to JAX arrays.
    - incomplete external field dictionaries fall back to zero arrays.
    - initial electric field x-component is produced from calculate_charge_density and E_from_Gauss_1D_Cartesian.
    """
    domain = domain_parameters(length=4.0, number_grid_points=4)
    solver = solver_parameters(filter_passes=0, filter_strides=(1,))
    domain_state = build_domain_state(domain)
    particle_state = {
        "positions": jnp.array([
            [-1.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]),
        "charges": jnp.array([
            [elementary_charge],
            [-elementary_charge],
        ]),
    }
    external_defaults = clean_and_initialize_external_field_parameters({})

    field_state = initialize_field_state(
        domain,
        solver,
        external_defaults,
        domain_state,
        particle_state,
    )

    E_field, B_field = field_state["fields"]
    charge_density = calculate_charge_density(
        particle_state["positions"],
        particle_state["charges"],
        domain_state["dx"],
        domain_state["grid"],
        domain["particle_BC_left"],
        domain["particle_BC_right"],
        solver["filter_passes"],
        solver["filter_alpha"],
        solver["filter_strides"],
        field_BC_left=domain["field_BC_left"],
        field_BC_right=domain["field_BC_right"],
    )
    expected_E_x = E_from_Gauss_1D_Cartesian(charge_density, domain_state["dx"])

    assert E_field.shape == (4, 3)
    assert B_field.shape == (4, 3)
    assert jnp.allclose(E_field[:, 0], expected_E_x)
    assert jnp.allclose(E_field[:, 1:], 0.0)
    assert jnp.allclose(B_field, 0.0)
    assert jnp.allclose(field_state["external_electric_field"], jnp.zeros((4, 3)))
    assert jnp.allclose(field_state["external_magnetic_field"], jnp.zeros((4, 3)))

    E_external = np.arange(12, dtype=np.float32).reshape(4, 3)
    B_external = -E_external
    external_fields = clean_and_initialize_external_field_parameters({
        "external_electric_field": {"E": E_external},
        "external_magnetic_field": {"B": B_external},
    })
    provided_field_state = initialize_field_state(
        domain,
        solver,
        external_fields,
        domain_state,
        particle_state,
    )

    assert provided_field_state["external_electric_field"].dtype == jnp.float32
    assert provided_field_state["external_magnetic_field"].dtype == jnp.float32
    assert jnp.allclose(provided_field_state["external_electric_field"], E_external)
    assert jnp.allclose(provided_field_state["external_magnetic_field"], B_external)

    incomplete_external_fields = clean_and_initialize_external_field_parameters({
        "external_electric_field": {},
        "external_magnetic_field": {},
    })
    fallback_field_state = initialize_field_state(
        domain,
        solver,
        incomplete_external_fields,
        domain_state,
        particle_state,
    )

    assert jnp.allclose(fallback_field_state["external_electric_field"], jnp.zeros((4, 3)))
    assert jnp.allclose(fallback_field_state["external_magnetic_field"], jnp.zeros((4, 3)))
