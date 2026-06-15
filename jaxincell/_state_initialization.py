import jax.numpy as jnp
from jax import lax
from jax.random import PRNGKey, normal, uniform

from ._constants import (
    elementary_charge,
    epsilon_0,
    mass_electron,
    mass_proton,
    speed_of_light,
)
from ._fields import E_from_Gauss_1D_Cartesian
from ._parameters._species_definitions import SPECIES_AXES, SPECIES_TYPES
from ._sources import calculate_charge_density

__all__ = [
    "build_domain_state",
    "initialize_field_state",
    "initialize_particle_state",
    "initialize_species_phase_space",
    "make_particles_from_state",
]

def build_domain_state(domain_parameters):
    length = domain_parameters["length"]
    length_y = jnp.where(domain_parameters["length_y"] == 0, length, domain_parameters["length_y"])
    length_z = jnp.where(domain_parameters["length_z"] == 0, length, domain_parameters["length_z"])

    number_grid_points = domain_parameters["number_grid_points"]
    number_grid_points_y = domain_parameters["number_grid_points_y"]
    number_grid_points_z = domain_parameters["number_grid_points_z"]
    if number_grid_points_y == 0:
        number_grid_points_y = 3
    if number_grid_points_z == 0:
        number_grid_points_z = 3

    dx = length / number_grid_points
    grid = jnp.linspace(-length / 2 + dx / 2, length / 2 - dx / 2, number_grid_points)
    dt = domain_parameters["timestep_over_spatialstep_times_c"] * dx / speed_of_light

    return {
        "box_size": (length, length_y, length_z),
        "dx": dx,
        "dt": dt,
        "grid": grid,
    }

def initialize_species_phase_space(species, seed_position, seed_velocity, number_particles, box_size):
    positions = []
    velocities = []
    for axis_index, axis in enumerate(SPECIES_AXES):
        axis_positions = lax.cond(
            species[f"random_positions_{axis}"],
            lambda _: uniform(
                PRNGKey(seed_position + axis_index + 1),
                shape=(number_particles,),
                minval=-box_size[axis_index] / 2,
                maxval=box_size[axis_index] / 2,
            ),
            lambda _: jnp.linspace(-box_size[axis_index] / 2, box_size[axis_index] / 2, number_particles),
            operand=None,
        )
        perturbation_wavenumber = species[f"perturbation_wavenumber_{axis}"] * 2 * jnp.pi / box_size[axis_index]
        axis_positions += (
            species[f"perturbation_amplitude_{axis}"]
            * jnp.sin(perturbation_wavenumber * axis_positions)
        )

        axis_velocities = (
            species[f"vth_over_c_{axis}"]
            * speed_of_light
            / jnp.sqrt(2)
            * normal(PRNGKey(seed_velocity + axis_index + 4), shape=(number_particles,))
        )
        axis_velocities += species[f"drift_speed_{axis}"]
        if species[f"velocity_plus_minus_{axis}"]:
            axis_velocities *= (-1) ** jnp.arange(0, number_particles)

        positions.append(axis_positions)
        velocities.append(axis_velocities)

    return jnp.stack(positions, axis=1), jnp.stack(velocities, axis=1)

def make_particles_from_state(
    species_parameters,
    species_type,
    rng_index,
    domain_parameters,
    solver_parameters,
    domain_state,
    electron_reference,
):
    species = species_parameters
    if species_type == "electrons":
        mass = mass_electron
    elif species_type == "ions":
        mass = species["mass_over_proton_mass"] * mass_proton
    else:
        raise ValueError(f"Unknown species type {species_type!r}.")

    charge = species["charge_over_elementary_charge"] * elementary_charge
    charge_mass = charge / mass
    thermal_speeds_over_c = jnp.array([
        species[f"vth_over_c_{axis}"]
        for axis in SPECIES_AXES
    ])
    thermal_speeds = thermal_speeds_over_c * speed_of_light

    seed = solver_parameters["seed"]
    number_particles = species["number_pseudoparticles"]
    box_size = domain_state["box_size"]
    number_grid_points = domain_parameters["number_grid_points"]

    assert rng_index >= 0
    local_seed = seed + 12 + rng_index * 6
    seed_position = local_seed
    if species["seed_position_override"]:
        seed_position = species["seed_position"]
    seed_velocity = local_seed

    positions, velocities = initialize_species_phase_space(
        species,
        seed_position,
        seed_velocity,
        number_particles,
        box_size,
    )

    out = {
        "charge": charge,
        "mass": mass,
        "charge_mass": charge_mass,
        "positions": positions,
        "velocities": velocities,
    }

    if species_type == "electrons" and rng_index == 0:
        electron_reference = {
            "vth_electrons": jnp.max(thermal_speeds),
            "vth_electrons_over_c": jnp.max(thermal_speeds_over_c),
            "charge_electrons": charge,
        }
    if electron_reference is None:
        raise ValueError("Electron reference species must be initialized before ions.")

    vth_electrons_over_c = electron_reference["vth_electrons_over_c"]
    charge_electrons = electron_reference["charge_electrons"]

    Debye_length_per_dx = 1 / species["grid_points_per_Debye_length"]
    weight = (
        epsilon_0
        * mass_electron
        * speed_of_light**2
        / charge_electrons**2
        * number_grid_points**2
        / box_size[0]
        / (2 * number_particles)
        * vth_electrons_over_c**2
        / Debye_length_per_dx**2
    )
    weight = jnp.where(species["weight"] == 0, weight, species["weight"])
    out['weights'] = weight * jnp.ones((number_particles, 1))

    return out, electron_reference

def initialize_particle_state(species_parameters, domain_parameters, solver_parameters, domain_state):
    position_blocks = []
    velocity_blocks = []
    weight_blocks = []
    species_index = []

    unique_species_indices = []
    species_integer_index_blocks = []
    charge_by_species = []
    mass_by_species = []
    charge_mass_by_species = []
    electron_reference = None

    for species_type in SPECIES_TYPES:
        for rng_index, species_label in enumerate(species_parameters[species_type]):
            species = species_parameters[species_type][species_label]
            species_particle_state, electron_reference = make_particles_from_state(
                species_parameters=species,
                species_type=species_type,
                rng_index=rng_index,
                domain_parameters=domain_parameters,
                solver_parameters=solver_parameters,
                domain_state=domain_state,
                electron_reference=electron_reference,
            )

            number_particles = species["number_pseudoparticles"]
            species_identifier = f"{species_type}.{species_label}"
            species_integer_index = len(unique_species_indices)

            position_blocks.append(species_particle_state["positions"])
            velocity_blocks.append(species_particle_state["velocities"])
            weight_blocks.append(species_particle_state["weights"])
            species_index.extend([species_identifier] * number_particles)
            species_integer_index_blocks.append(
                jnp.full((number_particles,), species_integer_index, dtype=jnp.int32)
            )
            unique_species_indices.append(species_identifier)
            charge_by_species.append(species_particle_state["charge"])
            mass_by_species.append(species_particle_state["mass"])
            charge_mass_by_species.append(species_particle_state["charge_mass"])

    positions = jnp.concatenate(position_blocks, axis=0)
    velocities = jnp.concatenate(velocity_blocks, axis=0)
    weights = jnp.concatenate(weight_blocks, axis=0)
    species_integer_index = jnp.concatenate(species_integer_index_blocks, axis=0)

    integer_key_map = {
        species_identifier: integer_index
        for integer_index, species_identifier in enumerate(unique_species_indices)
    }
    charge_lookup = dict(zip(unique_species_indices, charge_by_species))
    mass_lookup = dict(zip(unique_species_indices, mass_by_species))
    charge_mass_lookup = dict(zip(unique_species_indices, charge_mass_by_species))

    charge_integer_lookup = jnp.array(charge_by_species)
    mass_integer_lookup = jnp.array(mass_by_species)
    charge_mass_integer_lookup = jnp.array(charge_mass_by_species)

    charges = charge_integer_lookup[species_integer_index].reshape((-1,1)) * weights
    masses = mass_integer_lookup[species_integer_index].reshape((-1,1)) * weights
    charge_to_mass_ratios = charge_mass_integer_lookup[species_integer_index].reshape((-1,1))

    speed_limit = 0.99 * speed_of_light
    velocities = jnp.where(jnp.abs(velocities) >= speed_limit, jnp.sign(velocities) * speed_limit, velocities)

    return {
        "positions": positions,
        "velocities": velocities,
        "weights": weights,
        "species_index": species_index,
        "unique_species_indices": unique_species_indices,
        "integer_key_map": integer_key_map,
        "charge_lookup": charge_lookup,
        "mass_lookup": mass_lookup,
        "charge_mass_lookup": charge_mass_lookup,
        "charge_integer_lookup": charge_integer_lookup,
        "mass_integer_lookup": mass_integer_lookup,
        "charge_mass_integer_lookup": charge_mass_integer_lookup,
        "charges": charges,
        "masses": masses,
        "charge_to_mass_ratios": charge_to_mass_ratios,
        "vth_electrons": electron_reference["vth_electrons"],
        "vth_electrons_over_c": electron_reference["vth_electrons_over_c"],
        "charge_electrons": electron_reference["charge_electrons"],
    }

def initialize_field_state(domain_parameters, solver_parameters, external_field_parameters, domain_state, particle_state):
    grid = domain_state["grid"]
    positions = particle_state["positions"]
    charges = particle_state["charges"]
    dx = domain_state["dx"]

    B_field = jnp.zeros((grid.size, 3))
    E_field = jnp.zeros((grid.size, 3))

    charge_density = calculate_charge_density(positions, charges, dx, grid, domain_parameters["particle_BC_left"], domain_parameters["particle_BC_right"],
                                            solver_parameters["filter_passes"], solver_parameters["filter_alpha"], solver_parameters["filter_strides"],
                                            field_BC_left=domain_parameters["field_BC_left"], field_BC_right=domain_parameters["field_BC_right"])
    E_field_x = E_from_Gauss_1D_Cartesian(charge_density, dx)
    E_field = jnp.stack((E_field_x, jnp.zeros_like(grid), jnp.zeros_like(grid)), axis=1)

    G = domain_parameters['number_grid_points']

    secB = external_field_parameters.get("external_magnetic_field")
    secE = external_field_parameters.get("external_electric_field")
    if isinstance(secB, dict) and "B" in secB:
        external_magnetic_field = jnp.asarray(secB["B"], dtype=jnp.float32)
    else:
        external_magnetic_field = jnp.zeros((G, 3), dtype=jnp.float32)

    if isinstance(secE, dict) and "E" in secE:
        external_electric_field = jnp.asarray(secE["E"], dtype=jnp.float32)
    else:
        external_electric_field = jnp.zeros((G, 3), dtype=jnp.float32)

    return {
        "fields": (E_field, B_field),
        "external_magnetic_field": external_magnetic_field,
        "external_electric_field": external_electric_field,
    }
