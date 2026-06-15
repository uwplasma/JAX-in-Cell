import jax.numpy as jnp
from copy import deepcopy
from functools import partial
from jax_tqdm import scan_tqdm
from jax import lax, jit, config

from ._boundary_conditions import set_BC_positions, set_BC_particles
from ._algorithms import Boris_step, CN_step
from ._utils import make_differentiable_type
from ._parameters._sections import (
    DIFFERENTIABLE_INPUT_PARAMETERS,
    FLAT_DIFFERENTIABLE_INPUT_PARAMETERS,
    PARAMETER_SECTION_KEYS,
    PARAMETER_SECTIONS,
)
from ._parameters._species_definitions import (
    SPECIES_TYPES,
)
from ._parameters._species_parameters import resolve_species_references
from ._routing import (
    build_runtime_flat_parameter_routes,
    build_runtime_parameter_sections,
    build_runtime_species_label_routes,
    clean_runtime_input_parameters,
    route_nested_initial_species_parameters,
)
from ._state_initialization import build_domain_state, initialize_field_state, initialize_particle_state

try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib

config.update("jax_enable_x64", True)

__all__ = ["Simulation", "load_parameters"]

def load_parameters(input_file):
    parameters = tomllib.load(open(input_file, "rb"))
    return parameters

class Simulation:
    def __init__(self, parameters={}):
        if type(parameters) != dict:
            parameters = load_parameters(parameters)
        self.clean_and_initialize_parameters(parameters)
        self.reinitialize_simulation_state()
    
    def clean_and_initialize_parameters(self, parameters):
        input_parameters, parameters = self.classify_and_sort_input_parameters(parameters)

        parameter_sections = {
            section_name: parameters.pop(section_name, {})
            for section_name in PARAMETER_SECTIONS
        }

        input_parameters = {**parameters, **input_parameters}
        self._base_parameter_sections = deepcopy(parameter_sections)

        for section_name, section_metadata in PARAMETER_SECTIONS.items():
            setattr(
                self,
                section_metadata["attribute"],
                section_metadata["cleaner"](
                    parameter_sections[section_name],
                    input_parameters=input_parameters,
                ),
            )
    
    def classify_and_sort_input_parameters(self, parameters):
        """
        Sort through input parameters to move parameters into their respective dictionaries to overwrite defaults and
        move differentiable parameters into a separate input_parameters dictionary. This input_parameters dictionary
        can then be accessed to use them as inputs to the simulation function without having to write multiple
        toml files for the differentiable inputs and the non-differentiable parameters.
        """
        parameters = deepcopy(parameters)
        input_parameters = parameters.pop("input_parameters", {})
        differentiable_parameters = {}
        cleaner_input_parameters = {}
        unrouted_input_parameters = {}

        route_nested_initial_species_parameters(
            input_parameters,
            parameters,
            differentiable_parameters,
            cleaner_input_parameters,
        )

        for key, value in input_parameters.items():
            if key in SPECIES_TYPES:
                continue
            if key in FLAT_DIFFERENTIABLE_INPUT_PARAMETERS:
                differentiable_parameters[key] = make_differentiable_type(value)
                cleaner_input_parameters[key] = value
                continue

            routed_parameter = False
            for section_name, section_keys in PARAMETER_SECTION_KEYS.items():
                if key in section_keys:
                    parameters.setdefault(section_name, {})[key] = value
                    routed_parameter = True
                    break

            if not routed_parameter:
                cleaner_input_parameters[key] = value
                unrouted_input_parameters[key] = value

        self._input_parameters = differentiable_parameters
        self._unrouted_input_parameters = unrouted_input_parameters
        self.differentiable_input_parameters = DIFFERENTIABLE_INPUT_PARAMETERS

        return cleaner_input_parameters, parameters
    
    def build_hash_values(self):
        for section_metadata in PARAMETER_SECTIONS.values():
            setattr(
                self,
                section_metadata["hash_attribute"],
                section_metadata["hasher"](getattr(self, section_metadata["attribute"])),
            )

    def reinitialize_simulation_state(self):
        self._runtime_flat_parameter_routes = build_runtime_flat_parameter_routes(self._species_parameters)
        self._runtime_species_label_routes = build_runtime_species_label_routes(self._species_parameters)
        self.build_domain()
        self.initialize_particles()
        self.initialize_fields()
        self.build_hash_values()

    def clean_runtime_input_parameters(self, input_parameters=None):
        return clean_runtime_input_parameters(
            input_parameters,
            self._runtime_flat_parameter_routes,
            self._runtime_species_label_routes,
            self._species_parameters,
        )

    def current_domain_state(self):
        return {
            "box_size": self.box_size,
            "dx": self.dx,
            "dt": self.dt,
            "grid": self.grid,
        }

    def build_domain(self):
        domain_state = build_domain_state(self._domain_parameters)
        self.box_size = domain_state["box_size"]
        self.dx = domain_state["dx"]
        self.dt = domain_state["dt"]
        self.grid = domain_state["grid"]

    def initialize_particles(self):
        domain_state = self.current_domain_state()
        particle_state = initialize_particle_state(
            self._species_parameters,
            self._domain_parameters,
            self._solver_parameters,
            domain_state,
        )
        for key, value in particle_state.items():
            setattr(self, key, value)

    def initialize_fields(self):
        domain_state = self.current_domain_state()
        particle_state = {
            "positions": self.positions,
            "charges": self.charges,
        }
        field_state = initialize_field_state(
            self._domain_parameters,
            self._solver_parameters,
            self._external_field_parameters,
            domain_state,
            particle_state,
        )
        self.fields = field_state["fields"]
        self.external_magnetic_field = field_state["external_magnetic_field"]
        self.external_electric_field = field_state["external_electric_field"]

    @partial(jit, static_argnames=['self', 'domain_hash', 'species_hash', 'external_field_hash', 'source_hash', 'solver_hash'])
    def _simulation(self, input_parameters=None, domain_hash='', species_hash='', external_field_hash='', source_hash='', solver_hash=''):
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
        if input_parameters is None:
            input_parameters = {}

        base_parameter_sections = {
            section_name: getattr(self, section_metadata["attribute"])
            for section_name, section_metadata in PARAMETER_SECTIONS.items()
        }
        runtime_parameter_sections = build_runtime_parameter_sections(
            base_parameter_sections,
            input_parameters,
        )
        domain_parameters = runtime_parameter_sections["domain_parameters"]
        species_parameters = runtime_parameter_sections["species_parameters"]
        resolve_species_references(species_parameters)
        external_field_parameters = runtime_parameter_sections["external_field_parameters"]
        source_parameters = runtime_parameter_sections["source_parameters"]
        solver_parameters = runtime_parameter_sections["solver_parameters"]

        domain_state = build_domain_state(domain_parameters)
        particle_state = initialize_particle_state(
            species_parameters,
            domain_parameters,
            solver_parameters,
            domain_state,
        )
        field_state = initialize_field_state(
            domain_parameters,
            solver_parameters,
            external_field_parameters,
            domain_state,
            particle_state,
        )

        total_steps = domain_parameters["total_steps"]

        # Extract parameters for convenience
        dx = domain_state["dx"]
        dt = domain_state["dt"]
        grid = domain_state["grid"]
        box_size = domain_state["box_size"]
        E_field, B_field = field_state["fields"]
        charges = particle_state["charges"]
        masses = particle_state["masses"]
        charge_to_mass_ratios = particle_state["charge_to_mass_ratios"]
        field_BC_left = domain_parameters["field_BC_left"]
        field_BC_right = domain_parameters["field_BC_right"]
        particle_BC_left = domain_parameters["particle_BC_left"]
        particle_BC_right = domain_parameters["particle_BC_right"]

        # **Use provided positions/velocities if given, otherwise use defaults**
        # Need to add this functionality back in
        positions = None
        velocities = None
        if positions is None:
            positions = particle_state["positions"]
        if velocities is None:
            velocities = particle_state["velocities"]
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
            charges, masses, charge_to_mass_ratios,
            dx, grid, *box_size, particle_BC_left, particle_BC_right)

        positions_minus1_2 = set_BC_positions(
            positions - (dt / 2) * velocities,
            charges, dx, grid, *box_size,
            particle_BC_left, particle_BC_right)

        if solver_parameters["time_evolution_algorithm"] == 0:
            initial_carry = (
                E_field, B_field, positions_minus1_2, positions,
                positions_plus1_2, velocities, qs, ms, q_ms,
            )
            step_func = lambda carry, step_index: Boris_step(
                carry, step_index, solver_parameters, external_field_parameters, dx, dt, grid, box_size,
                particle_BC_left, particle_BC_right, field_BC_left, field_BC_right, solver_parameters['field_solver']
            )
        else:
            initial_carry = (
                E_field, B_field, positions,
                velocities, qs, ms, q_ms,
            )
            step_func = lambda carry, step_index: CN_step(
                carry, step_index, solver_parameters, dx, dt, grid, box_size,
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
        from ._constants import epsilon_0, mass_electron
        electron_species = next(iter(species_parameters["electrons"].values()))
        electron_weight = particle_state["weights"][0, 0]
        plasma_frequency = (
            jnp.sqrt(electron_species["number_pseudoparticles"] * electron_weight * particle_state["charge_electrons"]**2)
            / jnp.sqrt(mass_electron)
            / jnp.sqrt(epsilon_0)
            / jnp.sqrt(domain_parameters["length"])
        )
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
            "masses": masses,
            "charges": charges,
            "electric_field":  electric_field_over_time,
            "magnetic_field":  magnetic_field_over_time,
            "current_density": current_density_over_time,
            "charge_density":  charge_density_over_time,
            "number_grid_points":     domain_parameters["number_grid_points"],
            "number_pseudoelectrons": next(iter(species_parameters["electrons"].values()))["number_pseudoparticles"],
            "total_steps": total_steps,
            "time_array":  jnp.linspace(0, total_steps * dt, total_steps),
            "grid": grid,
            "dt": dt,
            "plasma_frequency": plasma_frequency,
            'dx': dx,
            'length': box_size[0],
            "external_electric_field": self.external_electric_field,
            "external_magnetic_field": self.external_magnetic_field,
        }

        # Might want to do more here
        output = {**temporary_output, **domain_parameters, **species_parameters, **external_field_parameters, **source_parameters, **solver_parameters}

        # diagnostics(output)

        return temporary_output
        #return output

    def simulation(self, input_parameters=None):
        if input_parameters is None:
            input_parameters = {}
        input_parameters = self.clean_runtime_input_parameters(input_parameters)
        return self._simulation(
            input_parameters,
            domain_hash=self.domain_hash,
            species_hash=self.species_hash,
            external_field_hash=self.external_field_hash,
            source_hash=self.source_hash,
            solver_hash=self.solver_hash,
        )
    
    def run(self, input_parameters=None):
        return self.simulation(input_parameters)

    def set_parameter_section(self, section_name, new_parameters):
        section_metadata = PARAMETER_SECTIONS[section_name]
        new_parameters = deepcopy(new_parameters)
        self._base_parameter_sections[section_name] = deepcopy(new_parameters)
        setattr(
            self,
            section_metadata["attribute"],
            section_metadata["cleaner"](new_parameters),
        )
        self.reinitialize_simulation_state()
    
    @property
    def domain_parameters(self):
        return self._domain_parameters
    
    @domain_parameters.setter
    def domain_parameters(self, new_domain_parameters):
        self.set_parameter_section("domain_parameters", new_domain_parameters)

    @property
    def species_parameters(self):
        return self._species_parameters
    
    @species_parameters.setter
    def species_parameters(self, new_species_parameters):
        self.set_parameter_section("species_parameters", new_species_parameters)
    
    @property
    def external_field_parameters(self):
        return self._external_field_parameters
    
    @external_field_parameters.setter
    def external_field_parameters(self, new_external_field_parameters):
        self.set_parameter_section("external_field_parameters", new_external_field_parameters)

    @property
    def source_parameters(self):
        return self._source_parameters
    
    @source_parameters.setter
    def source_parameters(self, new_source_parameters):
        self.set_parameter_section("source_parameters", new_source_parameters)
    
    @property
    def solver_parameters(self):
        return self._solver_parameters
    
    @solver_parameters.setter
    def solver_parameters(self, new_solver_parameters):
        self.set_parameter_section("solver_parameters", new_solver_parameters)
    
    @property
    def input_parameters(self):
        return deepcopy(self._input_parameters)
    
    @input_parameters.setter
    def input_parameters(self, new_input_parameters):
        parameters = deepcopy(self._base_parameter_sections)
        parameters["input_parameters"] = new_input_parameters
        self.clean_and_initialize_parameters(parameters)
        self.reinitialize_simulation_state()
