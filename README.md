<p align="center">
    <img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/refs/heads/main/docs/JAX-in-Cell_logo.png" align="center" width="30%">
</p>
<!-- <p align="center"><h3 align="center">JAX-IN-CELL</h1></p> -->
<p align="center">
	<em><code>❯ jaxincell: 1D3V particle-in-cell code in JAX to perform simulation of plasmas</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/uwplasma/JAX-in-Cell?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/uwplasma/JAX-in-Cell?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/uwplasma/JAX-in-Cell?style=default&color=0080ff" alt="repo-top-language">
	<a href="https://github.com/uwplasma/JAX-in-Cell/actions/workflows/build_test.yml">
		<img src="https://github.com/uwplasma/JAX-in-Cell/actions/workflows/build_test.yml/badge.svg" alt="Build Status">
	</a>
	<a href="https://codecov.io/gh/uwplasma/JAX-in-Cell">
		<img src="https://codecov.io/gh/uwplasma/JAX-in-Cell/branch/main/graph/badge.svg" alt="Coverage">
	</a>
	<a href="https://jax-in-cell.readthedocs.io/en/latest/?badge=latest">
		<img src="https://readthedocs.org/projects/jax-in-cell/badge/?version=latest" alt="Documentation Status">
	</a>

</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>


##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

JAX-in-Cell is an open-source project in Python that uses JAX to speedup simulations, leading to a simple to use, fast and concise code. It can be imported in a Python script using the **jaxincell** package, or run directly in the command line as `jaxincell`. To install it, use

   ```sh
   pip install jaxincell
   ```

Alternatively, you can install the Python dependencies `jax`, `jax_tqdm` and `matplotlib`, and run an [example script](examples/two-stream_instability.py) in the repository after downloading it as

   ```sh
   git clone https://github.com/uwplasma/JAX-in-Cell
   python examples/two-stream_instability.py
   ```

This allows JAX-in-Cell to be run without any installation.

An example without the use of an input toml file can be seen in the [Weibel instability example](examples/Weibel_instability.py)

The project can be downloaded in its [GitHub repository](https://github.com/uwplasma/JAX-in-Cell)
</code>

---

##  Features

JAX-in-Cell can run in CPUs, GPUs and TPUs, has autodifferentiation and just-in-time compilation capabilities, is based on rigorous testing, uses CI/CD via GitHub actions and has detailed documentation.

Currently, it evolves particles using the non-relativisic Lorentz force $\mathbf F = q (\mathbf E + \mathbf v \times \mathbf B)$, and evolves the electric $\mathbf E$ and magnetic $\mathbf B$ field using Maxwell's equations.

Plenty of examples are provided in the `examples` folder, and the documentation can be found in [Read the Docs](https://jax-in-cell.readthedocs.io/).

---

##  Project Structure

```sh
└── JAX-in-Cell/
    ├── LICENSE
    ├── docs
    ├── examples
    │   ├── Landau_damping.py
    │   ├── Langmuir_wave.py
    │   ├── ion-acoustic_wave.py
	│   ├── two-stream_instability.py
	│   ├── auto-differentiability.py
	│   ├── scaling_energy_time.py
    │   └── optimize_two_stream_saturation.py
    ├── jaxincell
    │   ├── algorithms.py
    │   ├── boundary_conditions.py
    │   ├── constants.py
    │   ├── diagnostics.py
    │   ├── filters.py
    │   ├── fields.py
    │   ├── particles.py
    │   ├── plot.py
    │   ├── simulation.py
    │   └── sources.py
    ├── main.py
    └── tests
        └── test_simulation.py
```

---
##  Getting Started

###  Prerequisites

- **Programming Language:** Python

Besides Python, JAX-in-Cell has minimum requirements. These are stated in [requirements.txt](requirements.txt), and consist of the Python libraries `jax`, `jax_tqdm` and `matplotlib`.

###  Installation

Install JAX-in-Cell using one of the following methods:

**Using PyPi:**

1. Install JAX-in-Cell from anywhere in the terminal:
```sh
pip install jaxincell
```

**Build from source:**

1. Clone the JAX-in-Cell repository:
```sh
git clone https://github.com/uwplasma/JAX-in-Cell
```

2. Navigate to the project directory:
```sh
cd JAX-in-Cell
```

3. Install the project dependencies:

```sh
pip install -r /path/to/requirements.txt
```

4. Install JAX-in-Cell:

```sh
pip install -e .
```

###  Usage
To run a simple case of JAX-in-Cell, you can simply call `jaxincell` from the terminal
```sh
jaxincell
```

This runs JAX-in-Cell using standard input parameters of Landau damping. To change input parameters, use a TOML file similar to the [example script](examples/input.toml) present in the repository as

```sh
jaxincell examples/input.toml
```

Additionally, it can be run inside a script, as shown in the [example script](examples/two-stream_instability.py) file
```sh
python examples/two-stream_instability.py
```

There, you can find most of the input parameters needed to run many test cases, as well as resolution parameters.

### Parameters

JAX-in-Cell is highly configurable. Below is a list of the available parameters that can be defined in the TOML configuration file or the Python input dictionary.

#### Solver Parameters

These parameters control the numerical discretization, algorithm selection, and simulation resolution.

<details>
<summary><strong>Click to expand full Solver Parameter Table</strong></summary>
<br>

| Parameter Key | Description |
| :--- | :--- |
| `number_grid_points` | Number of spatial grid cells. |
| `total_steps`  | Total number of time steps to run. |
| `number_pseudoelectrons` | Total number of electron macroparticles. |
| `number_pseudoparticles_species` | List of particle counts for additional species. |
| **Algorithms** | | |
| `time_evolution_algorithm` | **0**: Explicit Boris pusher<br> **1**: Implicit Crank-Nicolson |
| `field_solver` | **0**: Electromagnetic (Curl_EB)<br>**1**: Electrostatic (Gauss FFT)<br>**2**: Electrostatic (Gauss Finite Diff)<br>**3**: Poisson (FFT) |
| **Implicit Solver Settings** | | |
| `max_number_of_Picard_iterations_implicit_CN` | Max Picard iterations per time step (only for algorithm 1). |
| `number_of_particle_substeps_implicit_CN` | Number of particle orbit substeps (only for algorithm 1). |
</details>

#### Input Parameters

<details>
<summary><strong>Click to expand full Parameter Table</strong></summary>
<br>

| Parameter Key  | Description |
| :--- | :--- |
| `length` | Total length of the simulation box (meters). |
| `grid_points_per_Debye_length`  | $\Delta x$ over Debye length. |
| `timestep_over_spatialstep_times_c`  | CFL condition factor: $c \Delta t / \Delta x$. |
| **Initialization** | | |
| `seed` | Random seed for reproducibility. |
| `random_positions_x`  | Randomize particle positions in x axis. |
| `weight` | Particle weight. |
| **Species Properties** | | |
| `electron_charge_over_elementary_charge` | Electron charge (normalized to $e$). |
| `ion_charge_over_elementary_charge`  | Ion charge (normalized to $e$). |
| `ion_mass_over_proton_mass` | Ion mass (normalized to proton mass). |
| `relativistic` | Use relativistic Boris pusher. |
| `vth_electrons_over_c_x,y,z`  | Electron thermal velocity in x,y,z. |
| `ion_temperature_over_electron_temperature_x,y,z` | Ratio $T_i/T_e$ in x,y,z. |
| `electron_drift_speed_x` | Electron drift speed (m/s) in X. |
| `ion_drift_speed_x`  | Ion drift speed (m/s) in X. |
| `velocity_plus_minus_electrons_x` | Create counter-streaming electron populations in X. |
| **Perturbations** | | |
| `amplitude_perturbation_x` | Amplitude of density perturbation in X. |
| `wavenumber_electrons_x` | Mode number $k$ (factor of $2\pi/L$) for electrons in X. |
| **External Fields** | | |
| `external_electric_field_amplitude`  | Amplitude of external E-field (cosine). |
| `external_electric_field_wavenumber` | Wavenumber of external E-field. |
| `external_magnetic_field_amplitude`  | Amplitude of external B-field (cosine). |
| `external_magnetic_field_wavenumber` | Wavenumber of external B-field. |
| **Boundary Conditions** | | |
| `particle_BC_left,right` | Left,right particle boundary conditon (0: periodic, 1: reflective, 2: absorbing). |
| **Numerics** | | |
| `filter_passes` | Number of passes of the digital filter for charge/current density. |
| `filter_alpha`  | Smoothing strength (0.0 to 1.0). |
| `filter_strides` | Multi-scale filtering strides. |
| `tolerance_Picard_iterations_implicit_CN`  | Tolerance for implicit solver iterations. |
| `print_info` | Print simulation details to console. |
</details>

###  Bump on Tail Domenstration
Here is a simple demonstration with electrons and ions moving in the x direction, including additional species with different weights (i.e., different real-particle number densities). A detailed list of parameters can be found in the bump-on-tail example.

<h3>Periodic Boundary Condition</h3>
<video src="https://github.com/user-attachments/assets/5f085f92-cb65-4765-b586-19e727bd2aab" controls="controls" style="max-width: 100%;">
</video>

<br/>

<h3>Reflective Boundary Condition</h3>
<video src="https://github.com/user-attachments/assets/9f33bac8-319e-4aba-91fb-befc64bca70e" controls="controls" style="max-width: 100%;">
</video>

###  Testing
Run the test suite using the following command:
```sh
pytest .
```



---
##  Project Roadmap

- [X] **`Task 1`**: <strike>Run PIC simulation using several field solvers.</strike>
- [X] **`Task 2`**: <strike>Finalize example scripts and their documentation.</strike>
- [X] **`Task 3`**: <strike>Implement a relativistic equation of motion.</strike>
- [ ] **`Task 4`**: Implement collisions to allow the plasma to relax to a Maxwellian.
- [ ] **`Task 5`**: Implement guiding-center equations of motion.
- [X] **`Task 6`**: <strike>Implement an implicit time-stepping algorithm.</strike>
- [ ] **`Task 7`**: Generalize JAX-in-Cell to 2D.

---

##  Contributing

- **💬 [Join the Discussions](https://github.com/uwplasma/JAX-in-Cell/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/uwplasma/JAX-in-Cell/issues)**: Submit bugs found or log feature requests for the `JAX-in-Cell` project.
- **💡 [Submit Pull Requests](https://github.com/uwplasma/JAX-in-Cell/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/uwplasma/JAX-in-Cell
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/uwplasma/JAX-in-Cell/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=uwplasma/JAX-in-Cell">
   </a>
</p>
</details>

---

##  License

This project is protected under the MIT License. For more details, refer to the [LICENSE](LICENSE) file.

---

##  Acknowledgments

- This code was inspired by a previous implementation of a PIC code in JAX by Sean Lim [here](https://github.com/SeanLim2101/PiC-Code-Jax).
- We acknowledge the help of the whole [UWPlasma](https://rogerio.physics.wisc.edu/) plasma group.

---
