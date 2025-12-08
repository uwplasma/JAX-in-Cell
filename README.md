<p align="center">
    <img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/refs/heads/main/docs/JAX-in-Cell_logo.png" align="center" width="30%">
</p>
<!-- <p align="center"><h3 align="center">JAX-IN-CELL</h1></p> -->
<p align="center">
	<em><code>❯ jaxincell: 1D3V full electromagnetic (EM) particle-in-cell code in JAX to simulate plasmas</code></em>
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

JAX-in-Cell is an open-source 1D3V full electromagnetic (EM) particle-in-cell (PIC) code written in Python/JAX. It uses JAX for just-in-time compilation, automatic differentiation and GPU/TPU support, leading to a simple to use, fast and concise research code. It can be imported in a Python script using the **jaxincell** package, or run directly in the command line as `jaxincell`. To install it, use

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

**Full EM PIC.** By default, JAX-in-Cell advances particles with the (optionally relativistic) Lorentz force
$\mathbf F = q (\mathbf E + \mathbf v \times \mathbf B)$ and updates both the electric field $\mathbf E$ and
magnetic field $\mathbf B$ using the full Maxwell curl equations on a staggered 1D grid. This is a *full
electromagnetic PIC* scheme: it supports electrostatic and electromagnetic waves, beam instabilities
(two-stream, Weibel), and field propagation at the speed of light.

**Shape functions and digital filtering.** Charge and current are deposited with a quadratic (TSC) shape
function, and can be further smoothed by configurable digital filters applied to both charge density and
current density. The filter is controlled by three knobs in the input/TOML:
`filter_passes`, `filter_alpha`, and `filter_strides`, providing a standard Birdsall–Langdon-style way
to reduce grid-scale noise without changing the core EM PIC scheme.

In PIC terminology:

- **Full EM PIC** (what JAX-in-Cell implements by default) solves the coupled particle + Maxwell system
  (Ampère–Maxwell and Faraday laws) so that both $\mathbf E$ and $\mathbf B$ evolve in time.
- **Electrostatic PIC** would instead solve only Gauss/Poisson for $\mathbf E$ with $\mathbf B = 0$; this is
  effectively recovered in JAX-in-Cell by starting from $\mathbf B = 0$ and focusing on the longitudinal
  electric field.
- More advanced reduced models (Darwin, quasi-static, etc.) are not yet implemented; the code is designed
  so that such extensions can be added on top of the full EM formulation.


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
    │   ├── boundary_conditions.py
    │   ├── constants.py
    │   ├── diagnostics.py
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

This runs JAX-in-Cell using standard input parameters of Landau damping. To change input parameters, use a TOML file similar to the [example input](example_input.toml) present in the repository as

```sh
jaxincell example_input.toml
```

Additionally, it can be run inside a script, as shown in the [example script](example_script.py) file
```sh
python example_script.py
```

There, you can find most of the input parameters needed to run many test cases, as well as resolution parameters.
The `jaxincell` package has a single function `simulation()` that takes as arguments a dictionary input_parameters, the number of grid points, number of pseudoelectrons, total number of time steps, and the field solver to use.

In the [example script](example_script.py) file we write as inline comments the meaning of each input parameter.

#### Physics / numerical knobs

The main numerical “knobs” exposed by `simulation()` and the TOML input are:

- `field_solver` (int, in `solver_parameters`)

  During time-stepping, JAX-in-Cell **always** updates the fields via the full EM Maxwell curl equations.
  The `field_solver` flag controls how Gauss’s law
  \[
    \nabla\cdot \mathbf E = \rho / \varepsilon_0
  \]
  is enforced for the longitudinal electric field $E_x$ after each full EM update:

  - `field_solver = 0` — **No Gauss projection.** Pure Maxwell curl update (Yee-like). Gauss’s law is only
    satisfied to the accuracy of the charge/current deposition. Useful for quick tests, but Gauss and
    total-energy diagnostics may drift more.
  - `field_solver = 1` — **FFT-based Gauss projection (periodic).** After each field update, the code
    projects $E_x$ in Fourier space so that the *discrete* divergence (the same backward-difference used in
    the diagnostics) matches $\rho / \varepsilon_0$ on a periodic 1D grid. This is the recommended default
    for periodic Landau damping, Langmuir waves, and two-stream instability.
  - `field_solver = 2` — **Real-space (Cartesian) Gauss projection.** Enforces the same discrete Gauss law
    in physical space using a cumulative-sum solver. This is a good cross-check of the FFT projector and a
    template for more general boundary conditions.

- `time_evolution_algorithm` (int, in `solver_parameters`)

  - `time_evolution_algorithm = 0` — **Explicit EM PIC (default).** Leapfrog scheme with Boris pusher
    (non-relativistic or relativistic) and explicit Maxwell curl update. CFL-limited by
    $c \, \Delta t / \Delta x \lesssim 1$.
    
  - `time_evolution_algorithm = 1` — **Implicit Crank–Nicolson step (electrostatic field update).** Uses an
  implicit solve for the longitudinal electric field with particle substepping in time (no curlE/curlB EM
  update in this mode), useful as a prototype for stiff electrostatic problems.

- `relativistic` (bool, in `input_parameters`)

  - `False` — Use the standard non-relativistic Boris pusher.
  - `True` — Use a relativistic Boris pusher that advances particle momentum and enforces
    $|\mathbf v| < c$ consistently.

- `filter_passes`, `filter_alpha`, `filter_strides` (in `input_parameters`)

  Control a configurable digital filter applied to both the charge density and current density on the grid:

  - `filter_passes` — number of times the filter is applied (more passes ⇒ stronger smoothing).
  - `filter_alpha` — filter strength in each pass (typically between 0 and 1).
  - `filter_strides` — tuple of integer strides (e.g. `(1, 2, 4)`) for multi-scale smoothing.

  Together, these act as a Birdsall–Langdon-style low-pass filter on ρ and J, complementary to the
  built-in quadratic (TSC) particle shape functions.

- `species` and `number_pseudoparticles_species` (optional additional populations)

  In the TOML (or directly in `simulation(...)`) you can specify extra species under a `species` table,
  each with its own charge, mass, thermal velocities and drift speeds. The corresponding
  `number_pseudoparticles_species` tuple in `solver_parameters` controls how many macro-particles are
  used for each additional species.

Other important physics knobs include the thermal speeds (`vth_electrons_over_c_x,y,z`),
temperature ratios (`ion_temperature_over_electron_temperature_x,y,z`), and drift velocities
(`electron_drift_speed_*`, `ion_drift_speed_*`), which control the initial plasma state in the
provided example scripts (Landau damping, Langmuir waves, ion-acoustic waves, Weibel, etc.).

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
- [ ] **`Task 6`**: Implement an implicit time-stepping algorithm.
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
