import jax.numpy as jnp
from copy import deepcopy
from functools import partial
from jax_tqdm import scan_tqdm
from jax import lax, jit, config

from ._boundary_conditions import set_BC_positions, set_BC_particles
from ._algorithms import Boris_step, CN_step
from ._parameters._sections import (
    DIFFERENTIABLE_INPUT_PARAMETERS,
    PARAMETER_SECTIONS,
)
from ._parameters._species_parameters import resolve_species_references
from ._routing import (
    build_runtime_flat_parameter_routes,
    build_runtime_parameter_sections,
    build_runtime_species_label_routes,
    clean_runtime_input_parameters,
    route_flat_initial_parameters,
    route_nested_initial_species_parameters,
)
from ._state_initialization import (
    build_domain_state,
    initialize_field_state,
    initialize_particle_state,
    print_simulation_information,
)

try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib

config.update("jax_enable_x64", True)

__all__ = ["Simulation", "load_parameters"]

def load_parameters(input_file):
    """
        Load parameters from a given .toml input file given the path to the file.
    """
    parameters = tomllib.load(open(input_file, "rb"))
    return parameters

class Simulation:
    """
        Calling Simulation(parameters) will create a Simulation object with the provided parameters.
        The parameters should be provided as a dictionary, but can also be provided as a path to
        a .toml file containing the parameters. The parameters will be cleaned and initialized using
        the appropriate cleaner functions for each parameter section, and the simulation state will
        be initialized based on the cleaned parameters.

        The Simulation object will expose any differentiable parameters that were provided in
        input_parameters through Simulation_object.input_parameters. These input_parameters can then
        be passed to the simulation() or run() bound functions to run a simulation with these
        input_parameters exposed such that grads can be taken with respect to them. If simulation()
        or run() is called without input_parameters, it will use the parameters provided at initialization
        (which may include user-provided parameters and defaults) and will not overwrite any parameters
        with the input_parameters.

        Example without input parameters:

        sim = Simulation(parameters)
        simulation_output = sim.simulation()
        or
        simulation_output = sim.run()

        
        Example with input parameters:

        sim = Simulation(parameters)
        input_parameters = sim.input_parameters
        simulation_output = sim.simulation(input_parameters)
        or
        simulation_output = sim.run(input_parameters)

        
        Additionally,

        def scalar_objective(input_parameters):
            simulation_output = sim.run(input_parameters)
            return some_scalar_function_of(simulation_output)

        grad(scalar_objective)(input_parameters)

        will provide the gradient of some scalar objective function of the simulation output with respect to the input_parameters.
    """
    def __init__(self, parameters=None):
        if parameters is None:
            parameters = {}
        if type(parameters) != dict:
            parameters = load_parameters(parameters)
        self.clean_and_initialize_parameters(parameters)
        self.reinitialize_simulation_state()

    def simulation(self, input_parameters=None):
        """
            Exposed simulation call which doesn't expose the hash values for each section to prevent
            unintentionally forcing recompiles or not recompiling when necessary.
            
            input_parameters is a dictionary of differentiable parameters
            If simulation is called without input parameters, it will use
            the user input parameters or default parameters previously provided.
            input_parameters will overwrite any differentiable parameters 
            previously provided. This is meant to make it simple to expose grads
            derivatives with respect to the input parameters.
        """
        if input_parameters is None:
            input_parameters = {}
        input_parameters = self.clean_runtime_input_parameters(input_parameters)
        simulation_output = self._simulation(
            input_parameters,
            domain_hash=self.domain_hash,
            species_hash=self.species_hash,
            external_field_hash=self.external_field_hash,
            source_hash=self.source_hash,
            solver_hash=self.solver_hash,
        )
        return self.assemble_output(simulation_output, input_parameters)
     
    # See simulation(...) for details on the purpose of input_parameters.
    def run(self, input_parameters=None):
        return self.simulation(input_parameters)
    
    """
        domain_hash, species_hash, external_field_hash, source_hash, solver_hash
        are included as arguments here to ensure that changes to any of these hashes will trigger a recompilation of the
        simulation function with the new parameters. This is necessary because the simulation function is jitted and we
        want to make sure that it uses the most up-to-date parameters whenever it is called.
    """
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
        print_simulation_information(
            domain_parameters,
            species_parameters,
            external_field_parameters,
            solver_parameters,
            domain_state,
            particle_state,
        )
        field_state = initialize_field_state(
            domain_parameters,
            solver_parameters,
            external_field_parameters,
            domain_state,
            particle_state,
        )
        runtime_external_field_parameters = {
            **external_field_parameters,
            "external_electric_field": field_state["external_electric_field"],
            "external_magnetic_field": field_state["external_magnetic_field"],
        }

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

        positions = particle_state["positions"]
        velocities = particle_state["velocities"]

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
                carry, step_index, solver_parameters, runtime_external_field_parameters, dx, dt, grid, box_size,
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
            "charge_to_mass_ratios": charge_to_mass_ratios,
            "initial_positions": positions,
            "initial_velocities": velocities,
            "weights": particle_state["weights"],
            "species_integer_index": particle_state["species_integer_index"],
            "charge_integer_lookup": particle_state["charge_integer_lookup"],
            "mass_integer_lookup": particle_state["mass_integer_lookup"],
            "charge_mass_integer_lookup": particle_state["charge_mass_integer_lookup"],
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
            "max_initial_vth_electrons": particle_state["vth_electrons"],
            "vth_electrons_over_c": particle_state["vth_electrons_over_c"],
            "charge_electrons": particle_state["charge_electrons"],
            'dx': dx,
            'length': box_size[0],
            "box_size": box_size,
            "fields": field_state["fields"],
            "external_electric_field": field_state["external_electric_field"],
            "external_magnetic_field": field_state["external_magnetic_field"],
        }

        return temporary_output

    def assemble_output(self, simulation_output, input_parameters):
        base_parameter_sections = {
            section_name: getattr(self, section_metadata["attribute"])
            for section_name, section_metadata in PARAMETER_SECTIONS.items()
        }
        parameter_sections = build_runtime_parameter_sections(
            base_parameter_sections,
            input_parameters,
        )
        resolve_species_references(parameter_sections["species_parameters"])

        domain_parameters = parameter_sections["domain_parameters"]
        external_field_parameters = parameter_sections["external_field_parameters"]
        source_parameters = parameter_sections["source_parameters"]
        solver_parameters = parameter_sections["solver_parameters"]

        return {
            **domain_parameters,
            **external_field_parameters,
            **source_parameters,
            **solver_parameters,
            **simulation_output,
            "domain_parameters": domain_parameters,
            "species_parameters": parameter_sections["species_parameters"],
            "external_field_parameters": external_field_parameters,
            "source_parameters": source_parameters,
            "solver_parameters": solver_parameters,
            "parameter_sections": parameter_sections,
        }
    
    def clean_and_initialize_parameters(self, parameters):
        # Sort parameters to intended locations in parameters
        input_parameters, parameters = self.classify_and_sort_input_parameters(parameters)

        # Build initial structure of parameters with canonical section names
        parameter_sections = {
            section_name: parameters.pop(section_name, {})
            for section_name in PARAMETER_SECTIONS
        }

        input_parameters = {**parameters, **input_parameters}
        self._base_parameter_sections = deepcopy(parameter_sections)

        # Set the self. parameter sections of the Simulation object
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

        # Route species parameters to correct place in parameters dictionary
        route_nested_initial_species_parameters(
            input_parameters,
            parameters,
            differentiable_parameters,
            cleaner_input_parameters,
        )
        # Route non-species parameters to correct place in parameters dictionary
        unrouted_input_parameters = route_flat_initial_parameters(
            input_parameters,
            parameters,
            differentiable_parameters,
            cleaner_input_parameters,
        )

        # Separate differentiable parameters inserted into input_parameters in provided parameters
        # and expose them via Simulation_object.input_parameters for ease of use when passing to simulation(...) or run(...)
        self._input_parameters = differentiable_parameters
        self.differentiable_input_parameters = DIFFERENTIABLE_INPUT_PARAMETERS

        # Flag any unrecognized user provided parameters
        if unrouted_input_parameters:
            unrouted_keys = ", ".join(unrouted_input_parameters.keys())
            raise ValueError(
                "Initial input_parameters included parameter(s) that could not be routed. "
                f"Unrouted parameter(s): {unrouted_keys}"
            )

        return cleaner_input_parameters, parameters
    
    def build_hash_values(self):
        """
            Build the hashes for each of the parameter sections to help with determining when Jax
            needs to recompile the _simulation() function due to new parameters being passed.
        """
        for section_metadata in PARAMETER_SECTIONS.values():
            setattr(
                self,
                section_metadata["hash_attribute"],
                section_metadata["hasher"](getattr(self, section_metadata["attribute"])),
            )

    def reinitialize_simulation_state(self):
        """
            Reinitialize the simulation state based on the current parameter sections.
            This should be called whenever parameters are updated after initialization to
            ensure that the simulation state is consistent with the new parameters.
        """
        self._runtime_flat_parameter_routes = build_runtime_flat_parameter_routes()
        self._runtime_species_label_routes = build_runtime_species_label_routes(self._species_parameters)
        self.build_domain()
        self.initialize_particles()
        self.initialize_fields()
        self.build_hash_values()

    def clean_runtime_input_parameters(self, input_parameters=None):
        """
            Clean the input_parameters provided to the simulation(...) or run(...) functions at runtime.
        """
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

    def set_parameter_section(self, section_name, new_parameters):
        """
            Helper which is used by setters for the parameter sections to automatically
            clean parameters and reinitialize the state of the simulation including creating
            new hashes.
        """
        section_metadata = PARAMETER_SECTIONS[section_name]
        new_parameters = deepcopy(new_parameters)
        self._base_parameter_sections[section_name] = deepcopy(new_parameters)
        setattr(
            self,
            section_metadata["attribute"],
            section_metadata["cleaner"](new_parameters),
        )
        self.reinitialize_simulation_state()
    
    # Getters and setters from here on
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
