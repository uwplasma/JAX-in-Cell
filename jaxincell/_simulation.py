import jax.numpy as jnp
from functools import partial
from jax_tqdm import scan_tqdm
from jax import lax, jit, config
from jax.random import PRNGKey, uniform, normal

from ._sources import calculate_charge_density
from ._boundary_conditions import set_BC_positions, set_BC_particles
from ._constants import speed_of_light, epsilon_0, elementary_charge, mass_electron, mass_proton
from ._fields import E_from_Gauss_1D_Cartesian
from ._algorithms import Boris_step, CN_step
from ._utils import make_differentiable_type
from ._parameters import (
    ALL_DOMAIN_PARAMETERS,
    ALL_ELECTRON_PARAMETERS,
    ALL_EXTERNAL_FIELD_PARAMETERS,
    ALL_ION_PARAMETERS,
    ALL_LEGACY_SPECIES_PARAMETERS,
    ALL_SOLVER_PARAMETERS,
    ALL_SOURCE_PARAMETERS,
    DIFFERENTIABLE_DOMAIN_PARAMETERS,
    DIFFERENTIABLE_ELECTRON_PARAMETERS,
    DIFFERENTIABLE_EXTERNAL_FIELD_PARAMETERS,
    DIFFERENTIABLE_ION_PARAMETERS,
    DIFFERENTIABLE_LEGACY_SPECIES_PARAMETERS,
    DIFFERENTIABLE_SOLVER_PARAMETERS,
    DIFFERENTIABLE_SOURCE_PARAMETERS,
    DIFFERENTIABLE_SPECIES_PARAMETERS,
    build_domain_hash,
    build_external_field_hash,
    build_solver_hash,
    build_source_hash,
    build_species_hash,
    clean_and_initialize_domain_parameters,
    clean_and_initialize_external_field_parameters,
    clean_and_initialize_solver_parameters,
    clean_and_initialize_source_parameters,
    clean_and_initialize_species_parameters,
)

try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib

config.update("jax_enable_x64", True)

__all__ = ["Simulation", "load_parameters"]

PARAMETER_SECTION_KEYS = {
    "domain_parameters": ALL_DOMAIN_PARAMETERS,
    "species_parameters": ALL_LEGACY_SPECIES_PARAMETERS,
    "external_field_parameters": ALL_EXTERNAL_FIELD_PARAMETERS,
    "source_parameters": ALL_SOURCE_PARAMETERS,
    "solver_parameters": ALL_SOLVER_PARAMETERS,
}

MULTI_ROUTE_INPUT_PARAMETERS = {
    "grid_points_per_Debye_length": ("domain_parameters", "species_parameters"),
}

DIFFERENTIABLE_INPUT_PARAMETERS = list(dict.fromkeys(
    DIFFERENTIABLE_DOMAIN_PARAMETERS
    + DIFFERENTIABLE_LEGACY_SPECIES_PARAMETERS
    + DIFFERENTIABLE_SPECIES_PARAMETERS
    + DIFFERENTIABLE_EXTERNAL_FIELD_PARAMETERS
    + DIFFERENTIABLE_SOURCE_PARAMETERS
    + DIFFERENTIABLE_SOLVER_PARAMETERS
))

def copy_parameter_tree(value):
    if isinstance(value, dict):
        return {key: copy_parameter_tree(child) for key, child in value.items()}
    return value

def merge_parameter_trees(base_parameters, override_parameters):
    merged_parameters = copy_parameter_tree(base_parameters)
    for key, value in override_parameters.items():
        if isinstance(value, dict) and isinstance(merged_parameters.get(key), dict):
            merged_parameters[key] = merge_parameter_trees(merged_parameters[key], value)
        else:
            merged_parameters[key] = value
    return merged_parameters

def route_into_parameter_section(parameters, key, value):
    routed = False

    for section_name in MULTI_ROUTE_INPUT_PARAMETERS.get(key, ()):
        parameters.setdefault(section_name, {})[key] = value
        routed = True

    if routed:
        return True

    for section_name, section_keys in PARAMETER_SECTION_KEYS.items():
        if key in section_keys:
            parameters.setdefault(section_name, {})[key] = value
            return True

    return False

def species_parameter_is_differentiable(species_type, key):
    if species_type == "ions":
        return key in DIFFERENTIABLE_ION_PARAMETERS
    if species_type == "electrons":
        return key in DIFFERENTIABLE_ELECTRON_PARAMETERS
    return False

def species_parameter_is_known(species_type, key):
    if species_type == "ions":
        return key in ALL_ION_PARAMETERS
    if species_type == "electrons":
        return key in ALL_ELECTRON_PARAMETERS
    return False

def iter_species_parameter_groups(species_values):
    if not isinstance(species_values, dict):
        return []
    if any(isinstance(value, dict) for value in species_values.values()):
        return species_values.items()
    return [(None, species_values)]

def put_species_parameter(container, species_type, species_label, key, value):
    species_container = container.setdefault(species_type, {})
    if species_label is None:
        species_container[key] = value
    else:
        species_container.setdefault(species_label, {})[key] = value

def split_nested_species_input_parameters(input_parameters, parameters, differentiable_parameters, cleaner_input_parameters):
    for species_type in ("ions", "electrons"):
        for species_label, species_values in iter_species_parameter_groups(input_parameters.get(species_type, {})):
            for key, value in species_values.items():
                if species_parameter_is_differentiable(species_type, key):
                    put_species_parameter(differentiable_parameters, species_type, species_label, key, make_differentiable_type(value))
                    put_species_parameter(cleaner_input_parameters, species_type, species_label, key, value)
                elif species_parameter_is_known(species_type, key):
                    species_parameters = parameters.setdefault("species_parameters", {})
                    put_species_parameter(species_parameters, species_type, species_label, key, value)
                    put_species_parameter(cleaner_input_parameters, species_type, species_label, key, value)
                else:
                    put_species_parameter(cleaner_input_parameters, species_type, species_label, key, value)

def load_parameters(input_file):
    parameters = tomllib.load(open(input_file, "rb"))
    return parameters

'''
# Still need to rewrite this bit
def load_parameters(input_file):
    """
    Load parameters from a TOML file.

    Parameters:
    ----------
    input_file : str
        Path to the TOML file containing simulation parameters.

    Returns:
    -------
    parameters : dict
        Dictionary containing simulation parameters.
    """
    parameters = tomllib.load(open(input_file, "rb"))
    input_parameters = parameters['input_parameters']
    solver_parameters = parameters['solver_parameters']
    # Interface for additional species and/or particle populations
    try:
        # Nest within main struct to avoid changing top-level internal API
        input_parameters['species'] = parameters['species']
    except:
        input_parameters['species'] = []
    # Convert TOML array -> Python tuple to make hashable static argument, as
    # required by Jax
    try:
        solver_parameters['number_pseudoparticles_species'] = tuple(solver_parameters['number_pseudoparticles_species'])
    except KeyError:
        solver_parameters['number_pseudoparticles_species'] = ()
    if 'source_parameters' in parameters:
        pass # currently no source parameters, but are added in a separate branch, so this is a placeholder for future loading of source parameters
    return input_parameters, solver_parameters
'''

class Simulation:
    def __init__(self, parameters={}):
        if type(parameters) != dict:
            parameters = load_parameters(parameters)
        self.clean_and_initialize_parameters(parameters)
        self.reinitialize_simulation_state()

        # Still need to come up with a way to include this part, unfortunately
        '''
        # Compute derived parameters based on user-provided or default values
        for key, value in parameters.items():
            if callable(value):  # If the value is a lambda function, compute it
                parameters[key] = value(parameters)
        '''
    
    def clean_and_initialize_parameters(self, parameters):
        input_parameters, parameters = self.classify_and_sort_input_parameters(parameters)

        domain_parameters = parameters.pop("domain_parameters", {})
        species_parameters = parameters.pop("species_parameters", {})
        external_field_parameters = parameters.pop("external_field_parameters", {})
        source_parameters = parameters.pop("source_parameters", {})
        solver_parameters = parameters.pop("solver_parameters", {})

        input_parameters = {**parameters, **input_parameters}
        self._parameter_sections = copy_parameter_tree({
            "domain_parameters": domain_parameters,
            "species_parameters": species_parameters,
            "external_field_parameters": external_field_parameters,
            "source_parameters": source_parameters,
            "solver_parameters": solver_parameters,
        })

        self._domain_parameters = clean_and_initialize_domain_parameters(domain_parameters, input_parameters=input_parameters)
        self._species_parameters = clean_and_initialize_species_parameters(species_parameters, input_parameters=input_parameters)
        self._external_field_parameters = clean_and_initialize_external_field_parameters(external_field_parameters, input_parameters=input_parameters)
        self._source_parameters = clean_and_initialize_source_parameters(source_parameters, input_parameters=input_parameters)
        self._solver_parameters = clean_and_initialize_solver_parameters(solver_parameters, input_parameters=input_parameters)
    
    def classify_and_sort_input_parameters(self, parameters):
        """
        Sort through input parameters to move parameters into their respective dictionaries to overwrite defaults and
        move differentiable parameters into a separate input_parameters dictionary. This input_parameters dictionary
        can then be accessed to use them as inputs to the simulation function without having to write multiple
        toml files for the differentiable inputs and the non-differentiable parameters.
        """
        parameters = copy_parameter_tree(parameters)
        input_parameters = parameters.pop("input_parameters", {})
        differentiable_parameters = {}
        cleaner_input_parameters = {}
        unrouted_input_parameters = {}

        split_nested_species_input_parameters(
            input_parameters,
            parameters,
            differentiable_parameters,
            cleaner_input_parameters,
        )

        for key, value in input_parameters.items():
            if key in ("ions", "electrons"):
                continue
            if key in DIFFERENTIABLE_INPUT_PARAMETERS:
                differentiable_parameters[key] = make_differentiable_type(value)
                cleaner_input_parameters[key] = value
            elif not route_into_parameter_section(parameters, key, value):
                cleaner_input_parameters[key] = value
                unrouted_input_parameters[key] = value

        self._input_parameters = differentiable_parameters
        self._unrouted_input_parameters = unrouted_input_parameters
        self.differentiable_input_parameters = DIFFERENTIABLE_INPUT_PARAMETERS

        return cleaner_input_parameters, parameters
    
    def build_hash_values(self):
        self.domain_hash = build_domain_hash(self._domain_parameters)
        self.species_hash = build_species_hash(self._species_parameters)
        self.external_field_hash = build_external_field_hash(self._external_field_parameters)
        self.source_hash = build_source_hash(self._source_parameters)
        self.solver_hash = build_solver_hash(self._solver_parameters)

    def reinitialize_simulation_state(self):
        self.build_domain()
        self.initialize_particles()
        self.initialize_fields()
        self.build_hash_values()
    
    def build_domain(self):
        domain_parameters = self._domain_parameters

        length = domain_parameters["length"]
        length_y = domain_parameters["length_y"]
        length_z = domain_parameters["length_z"]
        if length_y == 0:
            length_y = length
        if length_z == 0:
            length_z = length
        
        number_grid_points = domain_parameters["number_grid_points"]
        number_grid_points_y = domain_parameters["number_grid_points_y"]
        number_grid_points_z = domain_parameters["number_grid_points_z"]
        if number_grid_points_y == 0:
            number_grid_points_y = 3
        if number_grid_points_z == 0:
            number_grid_points_z = 3
        
        self.box_size = (length, length_y, length_z)
        self.dx = length / number_grid_points
        dy = length_y / number_grid_points_y
        dz = length_z / number_grid_points_z

        self.grid = jnp.linspace(-length / 2 + self.dx / 2, length / 2 - self.dx / 2, number_grid_points)
        self.dt = domain_parameters["timestep_over_spatialstep_times_c"] * self.dx / speed_of_light

    def initialize_particles(self):
        species_parameters = self._species_parameters

        positions = jnp.zeros((0,3))
        velocities = jnp.zeros((0,3))
        weights = jnp.zeros((0,1))
        species_index = []

        charge_lookup = {}
        mass_lookup = {}
        charge_mass_lookup = {}

        # First the electrons
        for ii, species in enumerate(species_parameters["electrons"]):
            plists = self.make_particles(species_parameters = species_parameters['electrons'][species], species_type = "electrons", rng_index = ii)

            positions     = jnp.concatenate((positions, plists["positions"]))
            velocities    = jnp.concatenate((velocities, plists["velocities"]))
            weights       = jnp.concatenate((weights, plists["weights"]), axis=0)
            species_index = species_index + plists["species_index"]
            
            charge_lookup[plists["this_species_index"]]      = plists["charge"]
            mass_lookup[plists["this_species_index"]]        = plists["mass"]
            charge_mass_lookup[plists["this_species_index"]] = plists["charge_mass"]
        
        # Next the ions
        for ii, species in enumerate(species_parameters["ions"]):
            plists = self.make_particles(species_parameters = species_parameters['ions'][species], species_type = "ions", rng_index = ii)

            positions     = jnp.concatenate((positions, plists["positions"]))
            velocities    = jnp.concatenate((velocities, plists["velocities"]))
            weights       = jnp.concatenate((weights, plists["weights"]), axis=0)
            species_index = species_index + plists["species_index"]
            
            charge_lookup[plists["this_species_index"]]      = plists["charge"]
            mass_lookup[plists["this_species_index"]]        = plists["mass"]
            charge_mass_lookup[plists["this_species_index"]] = plists["charge_mass"]

        self.unique_species_indicies = set(species_index)
        self.integer_key_map = dict([(index, integer_index) for integer_index, index in enumerate(self.unique_species_indicies)])

        species_integer_index = jnp.array([self.integer_key_map.get(key) for key in species_index])

        charge_integer_lookup = [0] * len(self.unique_species_indicies)
        mass_integer_lookup = [0] * len(self.unique_species_indicies)
        charge_mass_integer_lookup = [0] * len(self.unique_species_indicies)
        for species in self.unique_species_indicies:
            charge_integer_lookup[self.integer_key_map[species]] = charge_lookup[species]
            mass_integer_lookup[self.integer_key_map[species]] = mass_lookup[species]
            charge_mass_integer_lookup[self.integer_key_map[species]] = charge_mass_lookup[species]
        
        self.charge_integer_lookup = jnp.array(charge_integer_lookup)
        self.mass_integer_lookup = jnp.array(mass_integer_lookup)
        self.charge_mass_integer_lookup = jnp.array(charge_mass_integer_lookup)
        
        # Keep this for now, can investigate tracking only weights and doing lookups later
        self.charges = self.charge_integer_lookup[species_integer_index].reshape((-1,1)) * weights
        self.masses = self.mass_integer_lookup[species_integer_index].reshape((-1,1)) * weights
        self.charge_to_mass_ratios = self.charge_mass_integer_lookup[species_integer_index].reshape((-1,1))
        
        speed_limit = 0.99 * speed_of_light
        velocities = jnp.where(jnp.abs(velocities) >= speed_limit, jnp.sign(velocities) * speed_limit, velocities)

        # Source particles (another branch) can go here or in their own dedicated function
        self.positions = positions
        self.velocities = velocities
        self.weights = weights
        self.species_index = species_index

        self.charge_lookup = charge_lookup
        self.mass_lookup = mass_lookup
        self.charge_mass_lookup = charge_mass_lookup
    
    def make_particles(self, species_parameters, species_type, rng_index):
        """
        Generate Nprt total particles of a user-requested species with specified
        charge, mass, and space/velocity distribution.

        Parameters:
        ----------
        species_parameters : dict
            Dictionary of user-specified species parameters.
        Nprt : int
            Total number of pseudoparticles in the domain
        box_size : tuple
            Domain size in x,y,z
        weight : float
            Top-level pseudoelectron weight
        seed : int
            Top-level random number generator seed used for entire simulation
        rng_index : int
            Species or particle population index in [0,1,2,3,...]
            Use a unique index value for each population.
            This index is used to advance the random seed and so avoid spurious
            correlation between different particle positions and velocities.
            See https://docs.jax.dev/en/latest/random-numbers.html

        Returns:
        -------
        plist : dict
            Dictionary with lists of positions, velocities, charges, masses.
        """
        _p = species_parameters
        if species_type == "electrons":
            mass = mass_electron
        elif species_type == "ions":
            mass = _p["mass_over_proton_mass"] * mass_proton
        charge = _p["charge_over_elementary_charge"] * elementary_charge
        charge_mass = charge / mass
        vth_x  = _p["vth_over_c_x"] * speed_of_light
        vth_y  = _p["vth_over_c_y"] * speed_of_light
        vth_z  = _p["vth_over_c_z"] * speed_of_light

        seed = self._solver_parameters["seed"]
        Nprt = _p["number_pseudoparticles"]
        box_size = self.box_size
        number_grid_points = self.domain_parameters['number_grid_points']

        # This code is brittle; it depends on hard-coded offsets to the RNG seed
        # within initialize_particles_fields(...)
        assert rng_index >= 0
        local_seed = seed+12 + rng_index*6
        # Separate position/velocity seeds allow different ion and electron
        # populations to be inited with identical space positions, but
        # uncorrelated velocity distributions
        seed_pos = local_seed
        if _p['seed_position_override']:
            seed_pos = _p['seed_position']
        seed_vel = local_seed

        out = dict()

        out['charge'] = charge
        out['mass'] = mass
        out['charge_mass'] = charge_mass

        # **Particle Positions**

        xs = lax.cond(_p["random_positions_x"],
            lambda _: uniform(PRNGKey(seed_pos+1), shape=(Nprt,),
                            minval=-box_size[0] / 2, maxval=box_size[0] / 2),
            lambda _: jnp.linspace(-box_size[0] / 2, box_size[0] / 2, Nprt), operand=None)
        perturbation_wavenumber_x = _p["perturbation_wavenumber_x"] * 2 * jnp.pi / box_size[0]
        xs += _p["perturbation_amplitude_x"] * jnp.sin(perturbation_wavenumber_x * xs)

        ys = lax.cond(_p["random_positions_y"],
            lambda _: uniform(PRNGKey(seed_pos+2), shape=(Nprt,),
                            minval=-box_size[1] / 2, maxval=box_size[1] / 2),
            lambda _: jnp.linspace(-box_size[1] / 2, box_size[1] / 2, Nprt), operand=None)
        perturbation_wavenumber_y = _p["perturbation_wavenumber_y"] * 2 * jnp.pi / box_size[1]
        ys += _p["perturbation_amplitude_y"] * jnp.sin(perturbation_wavenumber_y * ys)

        zs = lax.cond(_p["random_positions_z"],
            lambda _: uniform(PRNGKey(seed_pos+3), shape=(Nprt,),
                            minval=-box_size[2] / 2, maxval=box_size[2] / 2),
            lambda _: jnp.linspace(-box_size[2] / 2, box_size[2] / 2, Nprt), operand=None)
        perturbation_wavenumber_z = _p["perturbation_wavenumber_z"] * 2 * jnp.pi / box_size[2]
        zs += _p["perturbation_amplitude_z"] * jnp.sin(perturbation_wavenumber_z * zs)

        out['positions'] = jnp.stack((xs, ys, zs), axis=1)

        # **Particle Velocities**

        v_x = vth_x/jnp.sqrt(2) * normal(PRNGKey(seed_vel+4), shape=(Nprt,))
        v_y = vth_y/jnp.sqrt(2) * normal(PRNGKey(seed_vel+5), shape=(Nprt,))
        v_z = vth_z/jnp.sqrt(2) * normal(PRNGKey(seed_vel+6), shape=(Nprt,))
        v_x += _p["drift_speed_x"]
        v_y += _p["drift_speed_y"]
        v_z += _p["drift_speed_z"]
        if _p['velocity_plus_minus_x']:
            v_x *= (-1) ** jnp.arange(0, Nprt)
        if _p['velocity_plus_minus_y']:
            v_y *= (-1) ** jnp.arange(0, Nprt)
        if _p['velocity_plus_minus_z']:
            v_z *= (-1) ** jnp.arange(0, Nprt)

        out['velocities'] = jnp.stack((v_x, v_y, v_z), axis=1)

        # **Particle Species Labels and Weights**

        species_index = species_type + str(rng_index)
        out['this_species_index'] = species_index
        out['species_index'] = [species_index]*Nprt

        if species_index == 'electrons0':
            vth_electrons_over_c = jnp.max(jnp.array([vth_x, vth_y, vth_z]))
            self.vth_electrons = vth_electrons_over_c
            self.charge_electrons = charge
        else:
            vth_electrons_over_c = self.vth_electrons
        vth_electrons = vth_electrons_over_c * speed_of_light

        Debye_length_per_dx = 1 / _p["grid_points_per_Debye_length"]
        weight = (
            epsilon_0
            * mass_electron
            * speed_of_light**2
            / self.charge_electrons**2
            * number_grid_points**2
            / self.box_size[0]
            / (2 * Nprt)
            * vth_electrons_over_c**2
            / Debye_length_per_dx**2
        )
        print(weight)
        weight = jnp.where(_p["weight"]==0, weight, _p["weight"])
        Debye_length_per_dx = jnp.where(vth_electrons_over_c==0, 0, 1 / (jnp.sqrt(
                                weight
                                / epsilon_0
                                / mass_electron
                                * self.box_size[0]
                                * (2 * Nprt))
                                / speed_of_light
                                * (-self.charge_electrons)
                                / number_grid_points
                                / (vth_electrons_over_c)
        ))
        out['weights'] = weight * jnp.ones((Nprt, 1))

        # Could include weight_ratio again at a later point - not currently implemented
        #out['charges'] = charge * weight * _p['weight_ratio'] * jnp.ones((Nprt, 1))
        #out['masses']  = mass   * weight * _p['weight_ratio'] * jnp.ones((Nprt, 1))

        return out
    
    def initialize_fields(self):
        domain_parameters = self._domain_parameters
        solver_parameters = self._solver_parameters
        external_field_parameters = self._external_field_parameters
        grid = self.grid
        positions = self.positions
        charges = self.charges
        dx = self.dx

        B_field = jnp.zeros((grid.size, 3))
        E_field = jnp.zeros((grid.size, 3))
        
        charge_density = calculate_charge_density(positions, charges, dx, grid, domain_parameters["particle_BC_left"], domain_parameters["particle_BC_right"],
                                                solver_parameters["filter_passes"], solver_parameters["filter_alpha"], solver_parameters["filter_strides"],
                                                field_BC_left=domain_parameters["field_BC_left"], field_BC_right=domain_parameters["field_BC_right"])
        # Initial E field from charge density via Gauss's law
        E_field_x = E_from_Gauss_1D_Cartesian(charge_density, dx)
        E_field = jnp.stack((E_field_x, jnp.zeros_like(grid), jnp.zeros_like(grid)), axis=1)
        self.fields = (E_field, B_field)
        # --- External fields (arrays if provided at top-level; else zeros)
        G = self.domain_parameters['number_grid_points']

        secB = external_field_parameters.get("external_magnetic_field")
        secE = external_field_parameters.get("external_electric_field")
        # print(secB["B"])
        if isinstance(secB, dict) and "B" in secB:
            self.external_magnetic_field = jnp.asarray(secB["B"], dtype=jnp.float32)
        else:
            self.external_magnetic_field = jnp.zeros((G, 3), dtype=jnp.float32)

        if isinstance(secE, dict) and "E" in secE:
            self.external_electric_field = jnp.asarray(secE["E"], dtype=jnp.float32)
        else:
            self.external_electric_field = jnp.zeros((G, 3), dtype=jnp.float32)

        # external_E_field_x = parameters["external_electric_field_amplitude"] * jnp.cos(parameters["external_electric_field_wavenumber"] * jnp.linspace(-jnp.pi, jnp.pi, number_grid_points))
        # external_B_field_x = parameters["external_magnetic_field_amplitude"] * jnp.cos(parameters["external_magnetic_field_wavenumber"] * jnp.linspace(-jnp.pi, jnp.pi, number_grid_points))

    @partial(jit, static_argnames=['self', 'domain_hash', 'species_hash', 'external_field_hash', 'source_hash', 'solver_hash'])
    def _simulation(self, input_parameters={}, domain_hash='', species_hash='', external_field_hash='', source_hash='', solver_hash=''):
        """
        Run a plasma physics simulation using a Particle-In-Cell (PIC) method in JAX.

        This function simulates the evolution of a plasma system by solving for particle motion
        (electrons and ions) and self-consistent electromagnetic fields on a grid. It uses the
        Boris algorithm for particle updates and a leapfrog scheme for field updates.

        Parameters:
        ----------
        user_parameters : dict
            User-defined parameters for the simulation. These can include:
            - Physical parameters: box size, number of particles, thermal velocities.
            - Numerical parameters: grid resolution, time step size.
            - Boundary conditions for particles and fields.
            - Random seed for reproducibility.

        Returns:
        -------
        output : dict
        """
        # **Initialize simulation parameters**
        # do stuff with the input parameters to get it all into the right places

        domain_parameters = self._domain_parameters
        solver_parameters = self._solver_parameters
        total_steps = domain_parameters["total_steps"]

        # Extract parameters for convenience
        dx = self.dx
        dt = self.dt
        grid = self.grid
        box_size = self.box_size
        E_field, B_field = self.fields
        field_BC_left = domain_parameters["field_BC_left"]
        field_BC_right = domain_parameters["field_BC_right"]
        particle_BC_left = domain_parameters["particle_BC_left"]
        particle_BC_right = domain_parameters["particle_BC_right"]

        # **Use provided positions/velocities if given, otherwise use defaults**
        # Need to add this functionality back in
        positions = None
        velocities = None
        if positions is None:
            positions = self.positions
        if velocities is None:
            velocities = self.velocities
        '''
        # Ensure the provided positions/velocities match the expected shape
        if positions.shape != parameters["initial_positions"].shape:
            raise ValueError(f"Expected positions shape {parameters['initial_positions'].shape}, got {positions.shape}")
        if velocities.shape != parameters["initial_velocities"].shape:
            raise ValueError(f"Expected velocities shape {parameters['initial_velocities'].shape}, got {velocities.shape}")
        '''

        # Leapfrog integration: positions at half-step before the start
        positions_plus1_2, velocities, qs, ms, q_ms = set_BC_particles(
            positions + (dt / 2) * velocities, velocities,
            self.charges, self.masses, self.charge_to_mass_ratios,
            dx, grid, *box_size, particle_BC_left, particle_BC_right)

        positions_minus1_2 = set_BC_positions(
            positions - (dt / 2) * velocities,
            self.charges, dx, grid, *box_size,
            particle_BC_left, particle_BC_right)

        if solver_parameters["time_evolution_algorithm"] == 0:
            initial_carry = (
                E_field, B_field, positions_minus1_2, positions,
                positions_plus1_2, velocities, qs, ms, q_ms,
            )
            step_func = lambda carry, step_index: Boris_step(
                carry, step_index, solver_parameters, self._external_field_parameters, dx, dt, grid, box_size,
                particle_BC_left, particle_BC_right, field_BC_left, field_BC_right, solver_parameters['field_solver']
            )
        else:
            raise ValueError('CN is not yet implemented for the new set up')
            initial_carry = (
                E_field, B_field, positions,
                velocities, qs, ms, q_ms,
            )
            step_func = lambda carry, step_index: CN_step(
                carry, step_index, parameters, dx, dt, grid, box_size,
                particle_BC_left, particle_BC_right, field_BC_left, field_BC_right,
                solver_parameters["number_of_particle_substeps_implicit_CN"]
            )

        @scan_tqdm(total_steps)
        def simulation_step(carry, step_index):
            return step_func(carry, step_index)


        # Run simulation
        _, results = lax.scan(simulation_step, initial_carry, jnp.arange(total_steps))

        # Unpack results
        positions_over_time, velocities_over_time, electric_field_over_time, \
        magnetic_field_over_time, current_density_over_time, charge_density_over_time = results

        # **Output results**
        temporary_output = {
            ## segregate ions/electrons in non-jitted method outside simulation(...)
            ## so we can make use of dynamically constructed arrays
            #"position_electrons": positions_over_time[ :, :number_pseudoelectrons, :],
            #"velocity_electrons": velocities_over_time[:, :number_pseudoelectrons, :],
            #"mass_electrons":     parameters["masses"][   :number_pseudoelectrons],
            #"charge_electrons":   parameters["charges"][  :number_pseudoelectrons],
            #"position_ions":      positions_over_time[ :, number_pseudoelectrons:, :],
            #"velocity_ions":      velocities_over_time[:, number_pseudoelectrons:, :],
            #"mass_ions":          parameters["masses"][   number_pseudoelectrons:],
            #"charge_ions":        parameters["charges"][  number_pseudoelectrons:],
            "positions": positions_over_time,
            "velocities": velocities_over_time,
            "masses": self.masses,
            "charges": self.charges,
            "electric_field":  electric_field_over_time,
            "magnetic_field":  magnetic_field_over_time,
            "current_density": current_density_over_time,
            "charge_density":  charge_density_over_time,
            "number_grid_points":     domain_parameters["number_grid_points"],
            "number_pseudoelectrons": self.species_parameters["electrons"]['_electrons0']["number_pseudoparticles"],
            "total_steps": total_steps,
            "time_array":  jnp.linspace(0, total_steps * dt, total_steps),
            "grid": self.grid,
            "dt": self.dt,
            'plasma_frequency': 1e10,
            'dx': self.dx,
            'length': self.box_size[0]
        }

        # Might want to do more here
        output = {**temporary_output, **domain_parameters, **self.species_parameters, **self.external_field_parameters, **self.source_parameters, **solver_parameters}

        # diagnostics(output)

        return temporary_output
        #return output

    def rebuild_from_input_parameters(self, input_parameters):
        merged_input_parameters = merge_parameter_trees(self._input_parameters, input_parameters)
        parameters = copy_parameter_tree(self._parameter_sections)
        parameters["input_parameters"] = merged_input_parameters
        self.clean_and_initialize_parameters(parameters)
        self.reinitialize_simulation_state()

    def simulation(self, input_parameters={}):
        if input_parameters:
            self.rebuild_from_input_parameters(input_parameters)
        return self._simulation(
            input_parameters,
            domain_hash=self.domain_hash,
            species_hash=self.species_hash,
            external_field_hash=self.external_field_hash,
            source_hash=self.source_hash,
            solver_hash=self.solver_hash,
        )
    
    def run(self, input_parameters={}):
        return self.simulation(input_parameters)
    
    @property
    def domain_parameters(self):
        return self._domain_parameters
    
    @domain_parameters.setter
    def domain_parameters(self, new_domain_parameters):
        new_domain_parameters = copy_parameter_tree(new_domain_parameters)
        self._parameter_sections["domain_parameters"] = copy_parameter_tree(new_domain_parameters)
        self._domain_parameters = clean_and_initialize_domain_parameters(new_domain_parameters)
        self.reinitialize_simulation_state()

    @property
    def species_parameters(self):
        return self._species_parameters
    
    @species_parameters.setter
    def species_parameters(self, new_species_parameters):
        new_species_parameters = copy_parameter_tree(new_species_parameters)
        self._parameter_sections["species_parameters"] = copy_parameter_tree(new_species_parameters)
        self._species_parameters = clean_and_initialize_species_parameters(new_species_parameters)
        self.reinitialize_simulation_state()
    
    @property
    def external_field_parameters(self):
        return self._external_field_parameters
    
    @external_field_parameters.setter
    def external_field_parameters(self, new_external_field_parameters):
        new_external_field_parameters = copy_parameter_tree(new_external_field_parameters)
        self._parameter_sections["external_field_parameters"] = copy_parameter_tree(new_external_field_parameters)
        self._external_field_parameters = clean_and_initialize_external_field_parameters(new_external_field_parameters)
        self.reinitialize_simulation_state()

    @property
    def source_parameters(self):
        return self._source_parameters
    
    @source_parameters.setter
    def source_parameters(self, new_source_parameters):
        new_source_parameters = copy_parameter_tree(new_source_parameters)
        self._parameter_sections["source_parameters"] = copy_parameter_tree(new_source_parameters)
        self._source_parameters = clean_and_initialize_source_parameters(new_source_parameters)
        self.reinitialize_simulation_state()
    
    @property
    def solver_parameters(self):
        return self._solver_parameters
    
    @solver_parameters.setter
    def solver_parameters(self, new_solver_parameters):
        new_solver_parameters = copy_parameter_tree(new_solver_parameters)
        self._parameter_sections["solver_parameters"] = copy_parameter_tree(new_solver_parameters)
        self._solver_parameters = clean_and_initialize_solver_parameters(new_solver_parameters)
        self.reinitialize_simulation_state()
    
    @property
    def input_parameters(self):
        return self._input_parameters
    
    @input_parameters.setter
    def input_parameters(self, new_input_parameters):
        parameters = copy_parameter_tree(self._parameter_sections)
        parameters["input_parameters"] = new_input_parameters
        self.clean_and_initialize_parameters(parameters)
        self.reinitialize_simulation_state()
