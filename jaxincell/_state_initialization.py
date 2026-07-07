import jax.numpy as jnp
from jax import lax
from jax.debug import print as jprint
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
from ._parameters._species_parameters import coerce_species_initial_phase_space_parameters
from ._sources import calculate_charge_density
from ._boundary_conditions import field_2_ghost_cells

__all__ = [
    "build_domain_state",
    "initialize_field_state",
    "initialize_particle_state",
    "initialize_species_phase_space",
    "make_particles_from_state",
    "print_simulation_information",
]

def build_domain_state(domain_parameters):
    length = domain_parameters["length"]
    length_y = jnp.where(domain_parameters["length_y"] == 0, length, domain_parameters["length_y"])
    length_z = jnp.where(domain_parameters["length_z"] == 0, length, domain_parameters["length_z"])
    length_xyz = {'x': length, 'y': length_y, 'z': length_z}

    dimensions = ('x',)
    number_grid_points_xyz = {'x': domain_parameters["number_grid_points"]}
    number_grid_points_y = domain_parameters["number_grid_points_y"]
    number_grid_points_z = domain_parameters["number_grid_points_z"]

    if number_grid_points_y is not None and number_grid_points_y > 0:
        dimensions = dimensions + ('y',)
        number_grid_points_xyz['y'] = number_grid_points_y
    if number_grid_points_z is not None and number_grid_points_z > 0:
        dimensions = dimensions + ('z',)
        number_grid_points_xyz['z'] = number_grid_points_z

    dxyz = {}
    grid_xyz = {}
    for dim in dimensions:
        dxyz[dim] = length_xyz[dim] / number_grid_points_xyz[dim]
        grid_xyz[dim] = jnp.linspace(-length_xyz[dim] / 2 + dxyz[dim] / 2, length_xyz[dim] / 2 - dxyz[dim] / 2, number_grid_points_xyz[dim])

    dt = domain_parameters["timestep_over_spatialstep_times_c"] * dxyz['x'] / speed_of_light

    return {
        "box_size": (length, length_y, length_z),
        "dx": dxyz['x'],
        "dxyz": dxyz,
        "dt": dt,
        "grid": grid_xyz['x'],
        "grid_xyz": grid_xyz,
        "number_grid_points_xyz": number_grid_points_xyz,
        "dimensions": dimensions,
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

def species_seed_pair(seed, species_type, rng_index, extra_rng_index=None):
    if species_type == "electrons" and rng_index == 0:
        return seed, seed + 3
    if species_type == "ions" and rng_index == 0:
        return seed, seed + 6

    if extra_rng_index is None:
        extra_rng_index = max(rng_index - 1, 0)
    local_seed = seed + 12 + extra_rng_index * 6
    return local_seed, local_seed

def make_particles_from_state(
    species_parameters,
    species_type,
    rng_index,
    domain_parameters,
    solver_parameters,
    domain_state,
    electron_reference,
    seed_position=None,
    seed_velocity=None,
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
    if seed_position is None or seed_velocity is None:
        seed_position, seed_velocity = species_seed_pair(seed, species_type, rng_index)
    if species["seed_position_override"]:
        seed_position = species["seed_position"]

    positions, velocities = initialize_species_phase_space(
        species,
        seed_position,
        seed_velocity,
        number_particles,
        box_size,
    )
    coerce_species_initial_phase_space_parameters(
        species,
        f"{species_type}{rng_index}",
    )
    if species["initial_positions"] is not None:
        positions = species["initial_positions"]
    if species["initial_velocities"] is not None:
        velocities = species["initial_velocities"]

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
    extra_rng_index = 0

    for species_type in SPECIES_TYPES:
        for rng_index, species_label in enumerate(species_parameters[species_type]):
            species = species_parameters[species_type][species_label]
            seed_position, seed_velocity = species_seed_pair(
                solver_parameters["seed"],
                species_type,
                rng_index,
                extra_rng_index,
            )
            if not (rng_index == 0 and species_type in ("electrons", "ions")):
                extra_rng_index += 1
            species_particle_state, electron_reference = make_particles_from_state(
                species_parameters=species,
                species_type=species_type,
                rng_index=rng_index,
                domain_parameters=domain_parameters,
                solver_parameters=solver_parameters,
                domain_state=domain_state,
                electron_reference=electron_reference,
                seed_position=seed_position,
                seed_velocity=seed_velocity,
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
        "species_integer_index": species_integer_index,
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

def print_simulation_information(
    domain_parameters,
    species_parameters,
    external_field_parameters,
    solver_parameters,
    domain_state,
    particle_state,
):
    electron_species = next(iter(species_parameters["electrons"].values()))
    ion_species = next(iter(species_parameters["ions"].values()))

    length = domain_state["box_size"][0]
    dxyz = domain_state["dxyz"]
    dt = domain_state["dt"]
    total_steps = domain_parameters["total_steps"]
    number_grid_points = domain_parameters["number_grid_points"]
    number_pseudoelectrons = electron_species["number_pseudoparticles"]
    weight = particle_state["weights"][0, 0]
    charge_electrons = particle_state["charge_electrons"]
    vth_electrons = particle_state["vth_electrons"]
    Debye_length_per_dx = 1 / electron_species["grid_points_per_Debye_length"]
    electron_temperature = mass_electron * vth_electrons**2 / 2 / (-charge_electrons)
    plasma_frequency = (
        jnp.sqrt(number_pseudoelectrons * weight * charge_electrons**2)
        / jnp.sqrt(mass_electron)
        / jnp.sqrt(epsilon_0)
        / jnp.sqrt(length)
    )
    relativistic_gamma_factor = 1 / jnp.sqrt(
        1 - jnp.sum(particle_state["velocities"]**2, axis=1) / speed_of_light**2
    )

    lax.cond(
        solver_parameters["print_info"],
        lambda _: jprint((
            "Length of the simulation box: {} Debye lengths or {} Skin Depths\n"
            "Density of electrons: {} m^-3\n"
            "Electron temperature: {} eV\n"
            "Ion temperature / Electron temperature: {}\n"
            "Debye length: {} m\n"
            "Skin depth: {} m\n"
            "Wavenumber * Debye length: {}\n"
            "Pseudoparticles per cell: {}\n"
            "Pseudoparticle weight: {}\n"
            "Steps at each plasma frequency: {}\n"
            "Total time: {} / plasma frequency\n"
            "Number of particles on a Debye cube: {}\n"
            "Relativistic gamma factor: Maximum {}, Average {}\n"
            "Charge x External electric field x Debye Length / Temperature: {}\n"
        ),
            length / (Debye_length_per_dx * dxyz['x']),
            length / (speed_of_light / plasma_frequency),
            number_pseudoelectrons * weight / length,
            electron_temperature,
            ion_species["ion_temperature_over_electron_temperature_x"],
            Debye_length_per_dx * dxyz['x'],
            speed_of_light / plasma_frequency,
            electron_species["perturbation_wavenumber_x"] * Debye_length_per_dx * dxyz['x'],
            number_pseudoelectrons / number_grid_points,
            weight,
            1 / (plasma_frequency * dt),
            dt * plasma_frequency * total_steps,
            number_pseudoelectrons * weight / length * (Debye_length_per_dx * dxyz['x'])**3,
            jnp.max(relativistic_gamma_factor),
            jnp.mean(relativistic_gamma_factor),
            (
                -charge_electrons
                * external_field_parameters["external_electric_field_amplitude"]
                * Debye_length_per_dx
                * dxyz['x']
                / (mass_electron * vth_electrons**2 / 2)
            ),
        ),
        lambda _: None,
        operand=None,
    )

def initialize_field_state(domain_parameters, solver_parameters, external_field_parameters, domain_state, particle_state):
    grid_xyz = domain_state["grid_xyz"]
    positions = particle_state["positions"]
    charges = particle_state["charges"]
    dxyz = domain_state["dxyz"]
    grid = grid_xyz['x']

    B_field = jnp.zeros((grid.size, 3))
    E_field = jnp.zeros((grid.size, 3))

    charge_density = calculate_charge_density(positions, charges, dxyz['x'], grid, domain_parameters["particle_BC_left"], domain_parameters["particle_BC_right"],
                                            solver_parameters["filter_passes"], solver_parameters["filter_alpha"], solver_parameters["filter_strides"],
                                            field_BC_left=domain_parameters["field_BC_left"], field_BC_right=domain_parameters["field_BC_right"])
    E_field_x = E_from_Gauss_1D_Cartesian(charge_density, dxyz['x'])
    E_field = jnp.stack((E_field_x, jnp.zeros_like(grid), jnp.zeros_like(grid)), axis=1)

    G_xyz = domain_state['number_grid_points_xyz'].values()

    secB = external_field_parameters.get("external_magnetic_field")
    secE = external_field_parameters.get("external_electric_field")
    magnetic_field_input = secB.get("B") if isinstance(secB, dict) else secB
    electric_field_input = secE.get("E") if isinstance(secE, dict) else secE
    if magnetic_field_input is not None:
        external_magnetic_field = jnp.asarray(magnetic_field_input, dtype=jnp.float32)
    else:
        external_magnetic_field = jnp.zeros((*G_xyz, 3), dtype=jnp.float32)
    assert external_magnetic_field.shape == (*G_xyz, 3)

    if electric_field_input is not None:
        external_electric_field = jnp.asarray(electric_field_input, dtype=jnp.float32)
    else:
        external_electric_field = jnp.zeros((*G_xyz, 3), dtype=jnp.float32)
    assert external_electric_field.shape == (*G_xyz, 3)

    padded_external_electric_field, padded_external_magnetic_field = set_external_fields(
        external_electric_field,
        external_magnetic_field,
        domain_state["dimensions"],
    )

    return {
        "fields": (E_field, B_field),
        "external_magnetic_field": external_magnetic_field,
        "external_electric_field": external_electric_field,
        "padded_external_magnetic_field": padded_external_magnetic_field,
        "padded_external_electric_field": padded_external_electric_field,
    }

def add_ghost_cells_along_axis(field, axis, field_BC_left=0, field_BC_right=0):
    field = jnp.moveaxis(field, axis, 0)
    ghost_cell_L2, ghost_cell_L1, ghost_cell_R = field_2_ghost_cells(
        field_BC_left,
        field_BC_right,
        field,
    )

    field = jnp.concatenate(
        (
            ghost_cell_L2[jnp.newaxis, ...],
            ghost_cell_L1[jnp.newaxis, ...],
            field,
            ghost_cell_R[jnp.newaxis, ...],
        ),
        axis=0,
    )
    return jnp.moveaxis(field, 0, axis)

def set_external_fields(external_electric_field, external_magnetic_field, dimensions):
    """
    Set external fields based on user input and simulation dimensions.

    Parameters:
    ----------
    external_electric_field : jnp.ndarray
        User-provided external electric field array of shape (G, 3) or (G,).
    external_magnetic_field : jnp.ndarray
        User-provided external magnetic field array of shape (G, 3) or (G,).
    dimensions : tuple
        Tuple of dimension names ("x", "y", "z") present in the simulation.
    number_grid_pointsxyz : tuple
        Tuple of grid point counts in x, y, z directions.

    Returns:
    -------
    external_electric_field : jnp.ndarray
        External electric field array reshaped to match simulation grid.
    external_magnetic_field : jnp.ndarray
        External magnetic field array reshaped to match simulation grid.
    """
    for axis, _ in enumerate(dimensions):
        external_electric_field = add_ghost_cells_along_axis(external_electric_field, axis, field_BC_left=0, field_BC_right=0)
        external_magnetic_field = add_ghost_cells_along_axis(external_magnetic_field, axis, field_BC_left=0, field_BC_right=0)

    return external_electric_field, external_magnetic_field
