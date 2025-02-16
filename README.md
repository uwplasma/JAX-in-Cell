<p align="center">
    <img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/refs/heads/main/docs/JAX-in-Cell_logo.png" align="center" width="30%">
</p>
<!-- <p align="center"><h3 align="center">JAX-IN-CELL</h1></p> -->
<p align="center">
	<em><code>❯ jaxincell: 1D particle-in-cell code in JAX to perform simulation of plasmas</code></em>
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

Alternatively, you can install the Python dependencies `jax`, `jax_tqdm` and `matplotlib`, and run the [example script](example_script.py) in the repository after downloading it as

   ```sh
   git clone https://github.com/uwplasma/JAX-in-Cell
   python example_script.py
   ```

This allows JAX-in-Cell to be run without any installation.

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

###  Testing
Run the test suite using the following command:
```sh
pytest .
```

---
##  Project Roadmap

- [X] **`Task 1`**: <strike>Run PIC simulation using several field solvers.</strike>
- [ ] **`Task 2`**: Finalize example scripts and their documentation.
- [ ] **`Task 3`**: Implement a relativistic equation of motion.
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
