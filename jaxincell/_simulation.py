from copy import deepcopy
from functools import partial
from jax_tqdm import scan_tqdm
from jax import lax, jit, config, eval_shape, random

import jax.numpy as jnp
import numpy as np

from ._boundary_conditions import set_BC_positions, set_BC_particles
from ._algorithms import Boris_step, CN_step
from ._collisions import collide, coulomb_logarithm
from ._constants import mass_electron, elementary_charge, epsilon_0
from ._fields import E_from_Gauss_1D_Cartesian
from ._sources import calculate_charge_density
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
    """A particle-in-cell simulation: parameters, initial state and the time loop.

    The constructor takes a nested dictionary of parameters, or a path to a TOML
    file containing one. Each section is validated, the defaults are filled in,
    and the grid, the pseudo-particles and the initial fields are built. Nothing
    is compiled until :meth:`run` is called.

    Parameters that are floating-point physical inputs can be changed at run time
    without recompiling, and differentiated with respect to. They are exposed as
    ``Simulation.input_parameters`` and accepted as the argument of :meth:`run`.

    Args:
        parameters (dict or str or pathlib.Path, optional): The parameter tree, or
            a path to a TOML file. Defaults to the built-in configuration, which
            runs a two-stream instability.

    Example:
        Run with the parameters given at construction:

        .. code-block:: python

            sim = Simulation(parameters)
            output = sim.run()

        Re-run with a different drift speed, reusing the compiled program:

        .. code-block:: python

            output = sim.run({"electrons": {"electrons0": {"drift_speed_x": 7e7}}})

        Differentiate a scalar diagnostic with respect to that drift speed:

        .. code-block:: python

            import jax.numpy as jnp
            from jax import grad

            def mean_field(drift_speed):
                out = sim.run({"electrons": {"electrons0": {"drift_speed_x": drift_speed}}})
                return jnp.mean(out["electric_field"][:, :, 0])

            derivative = grad(mean_field)(6e7)

    Note:
        Reverse-mode differentiation works with the explicit integrator only; the
        implicit one uses a while loop, for which forward mode (``jax.jvp``,
        ``jax.jacfwd``) must be used instead.
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
            source_parameters,
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
            "padded_external_electric_field": field_state["padded_external_electric_field"],
            "padded_external_magnetic_field": field_state["padded_external_magnetic_field"],
        }

        total_steps = domain_parameters["total_steps"]

        # Extract parameters for convenience
        dxyz = domain_state["dxyz"]
        dx = dxyz['x']
        dt = domain_state["dt"]
        grid_xyz = domain_state["grid_xyz"]
        grid = grid_xyz['x']
        dimensions = domain_state["dimensions"]
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
        positions_plus1_2 = positions + dt/2*velocities
        qs, ms, q_ms = charges, masses, charge_to_mass_ratios
        if particle_BC_left == 0 and particle_BC_right == 0:
            positions_plus1_2, velocities, qs, ms, q_ms = set_BC_particles(
                positions_plus1_2, velocities, qs, ms, q_ms, dx, grid, *box_size, 0, 0)

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
                carry, step_index, solver_parameters, runtime_external_field_parameters, dxyz, dt, grid_xyz, box_size, dimensions,
                particle_BC_left, particle_BC_right, field_BC_left, field_BC_right, solver_parameters['field_solver'],
                **{key: domain_parameters[key] for key in
                   ("mixed_BC_weight", "COR_left", "COR_right", "mixed_BC_velocity_scale")},
                physical_masses=(particle_state["mass_integer_lookup"][particle_state["species_integer_index"]][:, None]
                                 if source_parameters["source_term_active"] else None),
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

        if self._solver_parameters["collisions"]:
            # Species remain contiguous in initialization order; q/m does not identify a population.
            blocks, start = [], 0
            for kind in ("electrons", "ions"):
                for species in species_parameters[kind].values():
                    count = species["number_pseudoparticles"]
                    blocks.append((start, count))
                    start += count
            pairs = tuple((a, b) for a in range(len(blocks)) for b in range(a, len(blocks)))
            weights = particle_state["weights"].reshape(-1)
            ids = particle_state["species_integer_index"]
            physical_mass = particle_state["mass_integer_lookup"][ids]
            physical_charge = particle_state["charge_integer_lookup"][ids]
            log = self._solver_parameters["coulomb_logarithm"]
            if log is None:
                ne = weights[:blocks[0][1]].sum() / domain_parameters["length"]
                te = mass_electron * particle_state["vth_electrons"] ** 2 / (2 * elementary_charge)
                log = coulomb_logarithm(ne, te)
            collision_key = random.PRNGKey(self._solver_parameters["seed"] or 0)

        changing_weights = particle_BC_left >= 2 or particle_BC_right >= 2
        collisionless_step = step_func
        def collision_step(carry, step_index):
            new_carry, output = collisionless_step(carry, step_index)
            if self._solver_parameters["collisions"]:
                # Scatter at integer time, then rebuild the following half drift.
                x, v = output[:2]
                v = collide(random.fold_in(collision_key, step_index), x, v, weights,
                            physical_mass, physical_charge, blocks, pairs, log,
                            dt, dx, domain_parameters["length"], grid.shape[0])
                half = set_BC_positions(x + (dt / 2) * v, new_carry[6], dx, grid,
                                        *box_size, particle_BC_left, particle_BC_right)
                new_carry = (*new_carry[:4], half, v, *new_carry[6:])
                output = (x, v, *output[2:])
            return new_carry, (*output, new_carry[7], new_carry[6]) if changing_weights else output
        step_func = collision_step

        sources = source_parameters["source_term_active"]
        if sources:
            initial_carry = initial_carry, jnp.zeros(17)

            def source_step(carry, step_index):
                state, budget = carry
                E, B, xm, x, xp, v, q, m, qm = state
                born = (particle_state["source_birth_steps"] == step_index)[:, None]
                xb, vb = particle_state["source_birth_positions"], particle_state["source_birth_velocities"]
                q = jnp.where(born, particle_state["nominal_charges"], q)
                m = jnp.where(born, particle_state["nominal_masses"], m)
                qm = jnp.where(born, particle_state["nominal_charge_to_mass_ratios"], qm)
                x, v = jnp.where(born, xb, x), jnp.where(born, vb, v)
                xm, xp = jnp.where(born, xb - dt/2*vb, xm), jnp.where(born, xb + dt/2*vb, xp)

                def birth_field(E):
                    rho = calculate_charge_density(x, q, dx, grid, particle_BC_left, particle_BC_right,
                                                   solver_parameters["filter_passes"], solver_parameters["filter_alpha"],
                                                   solver_parameters["filter_strides"], field_BC_left, field_BC_right)
                    Ex = E_from_Gauss_1D_Cartesian(rho, dx, periodic=field_BC_left == 0 and field_BC_right == 0)
                    return E.at[:, 0].set(Ex)

                E_born = lax.cond(jnp.any(born), birth_field, lambda E: E, E)
                births = jnp.concatenate((jnp.array([jnp.sum(born * particle_state["weights"]), jnp.sum(born * q),
                                                    jnp.sum(born * m * vb**2) / 2]),
                                         jnp.sum(born * m * vb, axis=0)))
                state, data = collisionless_step((E_born, B, xm, x, xp, v, q, m, qm), step_index)
                field_work = epsilon_0 * dx / 2 * jnp.sum(E_born[:, 0]**2 - E[:, 0]**2)
                budget += jnp.concatenate((births, data[-1], jnp.array([field_work])))
                return (state, budget), (*data[:-1], state[7], state[6], state[8], budget)

        step_func = source_step if sources else collision_step

        # Keep the requested snapshots in fixed-size buffers carried through the scan.
        # An unset snapshot_steps records every step.
        num_snapshots = len(self._snapshot_steps)
        snapshot_steps_arr = jnp.array(self._snapshot_steps, dtype=int)
        time_array = (snapshot_steps_arr + 1) * dt
        # The sentinel keeps the lookup valid after the last requested snapshot.
        snapshot_steps_with_end = jnp.array((*self._snapshot_steps, total_steps), dtype=int)

        _, sample_step_data = eval_shape(step_func, initial_carry, 0)
        snapshot_buffers = tuple(
            jnp.zeros((num_snapshots,) + leaf.shape, dtype=leaf.dtype)
            for leaf in sample_step_data
        )

        @scan_tqdm(total_steps)
        def scan_body(carry_and_buffers, step_index):
            sim_carry, buffers, next_snapshot = carry_and_buffers
            new_sim_carry, step_data = step_func(sim_carry, step_index)

            should_save = step_index == snapshot_steps_with_end[next_snapshot]

            def save_snapshot(buffers):
                return tuple(
                    buf.at[next_snapshot].set(value)
                    for buf, value in zip(buffers, step_data)
                )

            if num_snapshots:
                buffers = lax.cond(should_save, save_snapshot, lambda buffers: buffers, buffers)
            next_snapshot += should_save.astype(next_snapshot.dtype)
            # Returning None prevents scan from stacking a second output history.
            return (new_sim_carry, buffers, next_snapshot), None

        (final_carry, snapshot_buffers, _), _ = lax.scan(
            scan_body, (initial_carry, snapshot_buffers, jnp.array(0, dtype=int)), jnp.arange(total_steps)
        )
        positions_over_time, velocities_over_time, electric_field_over_time, \
        magnetic_field_over_time, current_density_over_time, charge_density_over_time, mus_over_time = snapshot_buffers[:7]

        # **Output results**
        if sources:
            final_carry, final_budget = final_carry

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
            "mus": mus_over_time,
            "number_grid_points":     domain_parameters["number_grid_points"],
            "number_pseudoelectrons": next(iter(species_parameters["electrons"].values()))["number_pseudoparticles"],
            "total_steps": total_steps,
            "time_array":  time_array,
            "final_state": {
                "time": total_steps * dt,
                "electric_field": final_carry[0], "magnetic_field": final_carry[1],
                "positions": final_carry[3 if solver_parameters["time_evolution_algorithm"] == 0 else 2],
                "velocities": final_carry[-4], "charges": final_carry[-3],
                "masses": final_carry[-2], "charge_to_mass_ratios": final_carry[-1],
            },
            "grid": grid,
            "grid_xyz": grid_xyz,
            "dt": dt,
            "plasma_frequency": plasma_frequency,
            "max_initial_vth_electrons": particle_state["vth_electrons"],
            "vth_electrons_over_c": particle_state["vth_electrons_over_c"],
            "charge_electrons": particle_state["charge_electrons"],
            'dx': dx,
            'dxyz': dxyz,
            'length': box_size[0],
            "box_size": box_size,
            "fields": field_state["fields"],
            "external_electric_field": field_state["external_electric_field"],
            "external_magnetic_field": field_state["external_magnetic_field"],
            "padded_external_electric_field": field_state["padded_external_electric_field"],
            "padded_external_magnetic_field": field_state["padded_external_magnetic_field"],
        }

        if changing_weights:
            temporary_output.update(masses_over_time=snapshot_buffers[7], charges_over_time=snapshot_buffers[8])
        if sources:
            physical_masses = particle_state["mass_integer_lookup"][particle_state["species_integer_index"]]
            temporary_output.update(
                masses=particle_state["nominal_masses"], charges=particle_state["nominal_charges"],
                charge_to_mass_ratios=particle_state["nominal_charge_to_mass_ratios"],
                masses_over_time=snapshot_buffers[7], charges_over_time=snapshot_buffers[8], charge_to_mass_ratios_over_time=snapshot_buffers[9],
                weights_over_time=snapshot_buffers[7][..., 0] / physical_masses,
                alive_particles=snapshot_buffers[7][..., 0] > 0, source_birth_steps=particle_state["source_birth_steps"],
                injected_weight=snapshot_buffers[10][:, 0], injected_charge=snapshot_buffers[10][:, 1], injected_energy=snapshot_buffers[10][:, 2],
                injected_momentum=snapshot_buffers[10][:, 3:6], lost_weight=snapshot_buffers[10][:, 6], lost_charge=snapshot_buffers[10][:, 7],
                lost_energy=snapshot_buffers[10][:, 8], lost_momentum=snapshot_buffers[10][:, 9:12], wall_energy_transfer=snapshot_buffers[10][:, 12],
                wall_momentum_transfer=snapshot_buffers[10][:, 13:16], source_field_work=snapshot_buffers[10][:, 16],
            )
            temporary_output["final_state"]["source_budget"] = final_budget
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
        if (self._solver_parameters["time_evolution_algorithm"] == 1 or self._solver_parameters["relativistic"]) and (
                any(self._domain_parameters[f"particle_BC_{side}"] >= 3 for side in ("left", "right"))
                or any(self._domain_parameters[f"COR_{side}"] != 1 for side in ("left", "right"))):
            raise ValueError("mixed or inelastic walls currently require nonrelativistic Boris")
        if self._solver_parameters["time_evolution_algorithm"] == 1:
            if self._solver_parameters["relativistic"] or any(
                self._domain_parameters[key] != 0
                for key in ("particle_BC_left", "particle_BC_right", "field_BC_left", "field_BC_right")
            ):
                raise ValueError("Implicit CN supports Newtonian particles with periodic particle and field boundaries only.")
        if self._solver_parameters["collisions"]:
            if self._source_parameters["source_term_active"]:
                raise ValueError("Coulomb collisions with particle sources are not yet validated.")
            assert self._domain_parameters["particle_BC_left"] == self._domain_parameters["particle_BC_right"] == 0, (
                "Coulomb collisions currently require periodic particle boundaries."
            )
        self._runtime_flat_parameter_routes = build_runtime_flat_parameter_routes()
        self._runtime_species_label_routes = build_runtime_species_label_routes(self._species_parameters)
        self.build_domain()
        self.resolve_snapshot_steps()
        self.initialize_particles()
        if any(self._domain_parameters[f"particle_BC_{side}"] >= 3 for side in ("left", "right")) and (
                self._solver_parameters["field_solver"] != 2):
            raise ValueError("mixed walls require field_solver=2 to account for collected charge")
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
            "dxyz": self.dxyz,
            "dt": self.dt,
            "grid": self.grid,
            "grid_xyz": self.grid_xyz,
            "number_grid_points_xyz": self.number_grid_points_xyz,
            "dimensions": self.dimensions,
        }

    def build_domain(self):
        domain_state = build_domain_state(self._domain_parameters)
        self.box_size = domain_state["box_size"]
        self.dx = domain_state["dx"]
        self.dxyz = domain_state["dxyz"]
        self.dt = domain_state["dt"]
        self.grid = domain_state["grid"]
        self.grid_xyz = domain_state["grid_xyz"]
        self.number_grid_points_xyz = domain_state["number_grid_points_xyz"]
        self.dimensions = domain_state["dimensions"]

    def resolve_snapshot_steps(self):
        snapshot_steps = self._solver_parameters["snapshot_steps"]
        if snapshot_steps is None:
            snapshot_steps = tuple(range(self._domain_parameters["total_steps"]))
        else:
            assert all(s < self._domain_parameters["total_steps"] for s in snapshot_steps), "Snapshot steps must be less than total_steps."
        self._snapshot_steps = snapshot_steps

    def initialize_particles(self):
        domain_state = self.current_domain_state()
        particle_state = initialize_particle_state(
            self._species_parameters,
            self._domain_parameters,
            self._solver_parameters,
            domain_state,
            self._source_parameters,
        )
        for key, value in particle_state.items():
            setattr(self, key, value)
        if not self._source_parameters["source_term_active"]:
            for key in ("source_birth_steps", "source_birth_positions", "source_birth_velocities",
                        "nominal_charges", "nominal_masses", "nominal_charge_to_mass_ratios"):
                self.__dict__.pop(key, None)

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
        if self._solver_parameters["time_evolution_algorithm"] == 1 and any(
                bool(np.any(np.asarray(field_state[name])))
                for name in ("external_electric_field", "external_magnetic_field")):
            raise ValueError("Implicit CN does not apply prescribed grid fields; use the explicit solver.")
        self.external_magnetic_field = field_state["external_magnetic_field"]
        self.external_electric_field = field_state["external_electric_field"]
        self.padded_external_magnetic_field = field_state["padded_external_magnetic_field"]
        self.padded_external_electric_field = field_state["padded_external_electric_field"]

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
        return deepcopy(self._domain_parameters)
    
    @domain_parameters.setter
    def domain_parameters(self, new_domain_parameters):
        self.set_parameter_section("domain_parameters", new_domain_parameters)

    @property
    def species_parameters(self):
        return deepcopy(self._species_parameters)
    
    @species_parameters.setter
    def species_parameters(self, new_species_parameters):
        self.set_parameter_section("species_parameters", new_species_parameters)
    
    @property
    def external_field_parameters(self):
        return deepcopy(self._external_field_parameters)
    
    @external_field_parameters.setter
    def external_field_parameters(self, new_external_field_parameters):
        self.set_parameter_section("external_field_parameters", new_external_field_parameters)

    @property
    def source_parameters(self):
        return deepcopy(self._source_parameters)
    
    @source_parameters.setter
    def source_parameters(self, new_source_parameters):
        self.set_parameter_section("source_parameters", new_source_parameters)
    
    @property
    def solver_parameters(self):
        return deepcopy(self._solver_parameters)
    
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
