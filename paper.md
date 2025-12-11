---
title: "JAX-in-Cell: A Differentiable Kinetic Plasma Code"
tags:
  - plasma physics
  - particle-in-cell
  - JAX
  - GPU acceleration
  - kinetic simulation
authors:
  - name: Longyu Ma
    affiliation: 1
  - name: Rogerio Jorge
    affiliation: 1
  - name: Hongke Lu
    affiliation: 1
  - name: Aaron Tran
    affiliation: 1
  - name: Christopher Woolford
    affiliation: 1
affiliations:
  - name: University of Wisconsin–Madison
    index: 1
date: July 2025
bibliography: paper.bib
---

# Summary
JAX-in-Cell is a fully electromagnetic, multispecies, and relativistic 1D3V Particle-in-Cell (PIC) framework implemented entirely in JAX. It provides a modern, Python-based alternative to traditional PIC frameworks. It leverages Just-In-Time (JIT) compilation and automatic vectorization to achieve the performance of traditional compiled codes on CPUs, GPUs, and TPUs. The resulting framework bridges the gap between educational scripts and production codes, providing a testbed for differentiable physics and AI integration that enables end-to-end gradient-based optimization. The code solves the Vlasov-Maxwell system on a staggered Yee lattice with either periodic, reflective, or absorbing boundary conditions, allowing both an explicit Boris solver and an implicit Crank-Nicolson method via Picard iteration to ensure energy conservation. Here, we detail the numerical methods employed, validate against standard benchmarks, and compare computational times in different hardware architectures, with and without JIT compilation.

# Statement of Need

A plasma is a collection of free ions and electrons whose self-generated and external electromagnetic fields drive collective behavior. Such behavior can be modelled using PIC simulations, which offer a fully kinetic description and enable exploration of complex interactions in modern plasma physics research, such as fusion devices, laser-plasma interactions, and astrophysical plasmas[@birdsall1991plasma].

However, such simulations can be computationally very expensive, often requiring hardware-specific implementations in low-level languages. The current landscape of high-performance production codes such as OSIRIS[@10.1007/3-540-47789-6_36], EPOCH[@Smith2021], VPIC[@10.1063/5.0146529], and WARPX[@WarpX], which are written in C++ or Fortran with MPI/CUDA backends, are highly optimized for massively parallel simulations, often with complex compilation chains, and require external adjoint implementations for optimization. This imposes a steep barrier to entry for new developers, making it cumbersome to test new algorithms, as well as to integrate with modern data science workflows.

JAX-in-Cell is able to fill this gap by implementing a 1D3V PIC framework entirely within the JAX ecosystem[@jax2018github]. It is open-source, user-friendly, and developer-friendly, written entirely in Python. It addresses three specific needs not met by existing codes: 1) hardware-agnostic high performance; 2) unified explicit and implicit solvers; 3) differentiable physics and AI integration. This is achieved using JAX's Just-In-Time (JIT) compilation via XLA, which allows us to achieve performance parity with compiled languages on CPUs, GPUs, and TPUs. Therefore, researchers can prototype new algorithms in Python and immediately use them on complex situations and accelerated hardware. Furthermore, JAX-in-Cell is inherently differentiable due to its automatic differentiation (AD) capabilities. This allows for new research directions such as optimization of laser pulse shapes, parameter discovery from experimental data, or embedding PIC simulations in Physics-Informed Neural Network loops. Finally, both an explicit scheme using the standard Boris algorithm[@boris1970relativistic] and a fully implicit, energy-conserving scheme[@chen2011energy] are available to cross-verify results and perform long simulations with large time steps, a capability often lacking in lightweight tools.


# Structure

Kinetic simulations can be performed using either a particle (or Lagrangian) approach, based on PIC simulations, or a continuum (or Eulerian) approach. While the former follows individual particles, these are down-sampled by following pseudo-particles, labeled $p$, leading to numerical noise that scales inversely with the square root of the mean number of particles[@birdsall1991plasma]. Nevertheless, PIC codes use an Eulerian grid for the fields and moments of the particle distribution function of species $s$, $f_s$, using therefore a mixed Eulerian/Lagrangian discretization. Particles are advanced along characteristics of the Vlasov equation
%
\begin{equation}
\partial_t f_s + \mathbf{v}\cdot\nabla f_s
+ \frac{q_s}{m_s} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B} \right)
\cdot \nabla_{\mathbf{u}} f_s = 0,
\end{equation}
%
with the electromagnetic fields governed by the standard Maxwell equations equations. Here, $\mathbf{v}$ is velocity, $q_s$ and $m_s$ are the
particle charge and mass, $\mathbf{E}$ and $\mathbf{B}$ are the electric and magnetic fields, $\mathbf{u} = \mathbf{v}\gamma$ is the
proper velocity, and $\gamma = \sqrt{1 + u^2/c^2}$ is the Lorentz factor, with $c$ the speed of light. The distribution function $f_s$ is discretized as
%
\begin{equation}
f_s(\mathbf{x}, \mathbf{v}) \approx \sum_{p} w_p\, \delta(\mathbf{x} - \mathbf{x_p})\, \delta(\mathbf{v} - \mathbf{v_p}),
\end{equation}
%
where $x_p$ denotes the position of each pseudo-particle, $\mathbf{v_p}$ denotes the velocity of each pseudo-particle, the weight is given by $w_p = n_0L/N$, with $n_0$ number density, $L$ the spatial domain length and $N$ the number of pseudo-particles for that species. Then, the spatial domain is divided into $N_x$ uniform cells with spacing $\Delta x$ and advanced in time by $\Delta t$. To mitigate numerical noise, each pseudo-particle is represented by a triangular shape function spanning three grid cells, and the same kernel is used consistently for both the particle-to-grid charge deposition and the grid-to-particle field interpolation[@hockney1988computer]. Accordingly, the current density $j$ is computed from the continuity equation using a discretely charge-conserving scheme[@villasenor1992rigorous] consistent with the shape function. Continuum Eulerian methods discretize the distribution function in six dimensions of phase space, making them computationally more costly, but, on the other hand, they are not subject to noise and can be cast in conservation law form so as to preserve the velocity moments of the distribution function.

The core logic of JAX-in-Cell is distributed along six specific modules. The first one is $\texttt{simulation.py}$, which serves as the high-level entry point, handling parameter initialization (via TOML parsing), memory allocation for particle and field arrays, and the execution of the main loop. The time-stepping is performed using a JAX primitive $\texttt{jax.lax.scan}$, which allows it to be optimized by the XLA compiler. Then, $\texttt{algorithms.py}$ implements the time integration schemes, which advance the system state by one timestep $\Delta t$ by a sequence of operations, namely particle push, source deposition, and field update. We implement two time-evolution methods (see Fig. autoref{fig:algorithm}), an explicit Boris algorithm[@boris1970relativistic], and an implicit Crank--Nicolson scheme solved via Picard iteration[@CHEN20142391]. The JIT-compiled particle dynamics is present in $\texttt{particles.py}$, which includes the relativistic and non-relativistic Boris rotation and the field interpolation routines, which are heavily vectorized using $\texttt{jax.vmap}$. The electromagnetic solvers are present in $\texttt{fields.py}$, which include the finite-difference curl operators for the Faraday and Ampère laws, as well as the divergence cleaning routines that enforce charge conservation. The deposition of charge and current densities from particle positions onto the grid is handled by $\texttt{sources.py}$, which implements high-order spline interpolation and digital filtering to mitigate aliasing noise. Finally, $\texttt{boundary\_conditions.py}$ is the centralized module to enforce boundary constraints, including routines for particle reflection/absorption and ghost-cell updates for the electromagnetic fields.

Due to its simplified design, JAX-in-Cell is able to pass the entire simulation state between functions, which is maintained as an immutable tuple referred to in the code as the $\texttt{carry}=(\mathbf E, \mathbf B, \mathbf x_p, \mathbf v_p, q, m)$.
This allows the entire simulation to be treated as a single differentiable function, which can facilitate the integration of automatic differentiation workflows.
The code can be run in two different ways: 1) directly called from the command line as $\texttt{jaxincell}$, pointing to a given TOML file, 2) from a Python driver script, either loading a TOML file using the \texttt{load\_parameters} function or using a dictionary of input parameters. While the code uses SI units internally, it allows initialization via dimensionless ratios common in plasma physics literature. Functions for diagnostics and plotting are available to compute and show phase-space structure, energy partition, and the evolution of the electromagnetic fields and sources.

![Time-stepping algorithms in JAX-in-Cell. Left: explicit Boris time-stepper and a Finite-Difference Time-Domain (FDTD) method using a staggered Yee grid for the electromagnetic fields. Right: implicit Crank-Nicolson time stepper using a Picard iteration for the electromagnetic system.\label{fig:algorithm}](figs/JAX-in-Cell_algorithm.pdf)

In order to reduce kernel launch overheads on GPUs, as well as the vector lengths for different populations, JAX-in-Cell adopts a monolithic array formulation of the multi-species architecture optimized for the Single-Instruction-Multiple-Data (SIMD) paradigm of JAX. That is, the simulation state, $\texttt{carry}$, is a single concatenated state of unified, global arrays, regardless of the number of physical species defined in the input configuration. This is a different approach than the one used in traditional PIC codes, which employ an object-oriented design with different species stored in separate containers and iterated over sequentially. During initialization, the code parses the TOML configuration file for the list of species and, for each population, the function $\texttt{make\_particles}$ generates each phase-space distribution and weights at the initial time, which are then appended to the global state vectors. This allows load balancing across GPU cores and minimizes warp divergence.



# Capabilities
The code is designed to simulate plasma dynamics, giving users full control over a wide range of physical and numerical parameters, including particle distributions, thermal velocities, drift speeds, external fields, number of particle species, and numerical tolerances. It also offers flexible options for initializing perturbations and multi-stream velocity configurations. Additionally, explicit algorithm also support relativistic particle dynamics and reflective or absorbing boundary conditions.

To facilitate analysis, the code automatically computes key quantities such as spatial scales (e.g., Debye length, skin depth), plasma frequency, and diagnostics of electric and magnetic field energies, as well as kinetic energies of electrons and ions to monitor energy conservation.

For example, the code can simulate phenomena such as Landau damping, the two-stream instability, and the Weibel instability. In these simulations, the plasma is neutralized by a uniform ion background and confined within a periodic domain of length $L$. The initial electron distribution with a position perturbation is given by

\begin{equation}
f_e(x, \mathbf{v}, t = 0) = f_{e0}(\mathbf{v})\left[ 1 + a \cos(kx)\right],
\end{equation}

where $a$ is the perturbation amplitude and $k$ the perturbation wavenumber and the velocities distribution reads

\begin{equation}
\begin{aligned}
f_{e0}(\mathbf{v}) &= \frac{n_0}{2 \, \pi^{3/2} v_{th,x} v_{th,y} v_{th,z}} 
\Bigg[ 
\exp\Big(- \frac{(v_x - v_{b1})^2}{v_{th,x}^2} - \frac{v_y^2}{v_{th,y}^2} - \frac{v_z^2}{v_{th,z}^2} \Big) \\
&\quad + 
\exp\Big(- \frac{(v_x - v_{b2})^2}{v_{th,x}^2} - \frac{v_y^2}{v_{th,y}^2} - \frac{v_z^2}{v_{th,z}^2} \Big)
\Bigg].
\end{aligned}
\end{equation}

where $v_{b1,2}$ are drift speeds along $x$ and $v_{th,i}$ is thermal velocities along each direction. To validate these simulations, it is useful to compare with the corresponding linear theory. Linearizing the Vlasov–Maxwell system around this initial distribution (with equal thermal velocities $v_{th,x}=v_{th,y}=v_{th,z}$), yields the dispersion relation:

\begin{equation}
1 + \frac{1}{2k^2\lambda_D^2}
\left[  2 + \xi_1 Z(\xi_1)+\xi_2 Z(\xi_2)\right] = 0, \quad
\xi_i=\frac{\omega}{kv_{th}}-\frac{v_{b_i}}{v_{th}},
\end{equation}

where $\lambda_D$ is Debye length and $Z$ is the Fried–Conte plasma dispersion function. The complex frequency $\omega$ determines both the oscillation frequency and the damping or growth rate $\gamma$. With this theoretical prediction established, the following parameter choices demonstrate two representative test cases using non-relativistic algorithms:

For Landau damping:
(i) Perturbation: $a = 0.025$, $k\lambda_D = 1/2$
(ii) Velocities: $v_{b_1} = v_{b_2} = 0$, $v_{th} = 0.35\,c$
(iii) Discretization: $N = 40{,}000$, $N_x = 32$, $\Delta x = 0.4\lambda_D$ and $\Delta t = 0.1\,\omega_{pe}^{-1}$

For two-stream instability:
(i) Perturbation: $a = 5\times10^{-7}$, $k\lambda_D = 1/8$
(ii) Velocities: $v_{b_1} = -v_{b_2} = 0.2\,c$, $v_{th} = 0.05\,c$
(iii) Discretization: $N = 10{,}000$, $N_x = 100$, $\Delta x = 0.5 \lambda_D$ and $\Delta t = 0.1\,\omega_{pe}^{-1}$

![Electric field energy evolution for Landau damping and two-stream instability. (a) Landau damping with analytical damping rate. (b) Two-stream instability showing fitted exponential growth rate. (c–d) Relative total energy deviation $|E_{\text{total}} - E_{\text{total}}(0)| / E_{\text{total}}(0)$ demonstrating energy conservation.\label{fig:output}](figs/output.png)

The results in \autoref{fig:output} show good agreement with analytical predictions; nevertheless, the Landau damping simulation exhibits high sensitivity to the initial conditions, in particular the choice of perturbation amplitude.

Next, we investigated the Weibel instability, which arises in anisotropic plasmas and leads to spontaneous magnetic field generation. The plasma is initialized with anisotropic velocity distribution, and we track magnetic field evolution. During the instability, the magnetic field organizes into filamentary structures perpendicular to the velocity anisotropy. Initially, multiple small filaments form, which subsequently merge into larger-scale structures as the system evolves (\autoref{fig:Weibel}).

![Evolution of the magnetic field during the Weibel instability. (a) Time evolution of total magnetic field energy. (b) Spatial profile of the magnetic field.\label{fig:Weibel}](figs/Weibel.png)

Additionally, to demonstrate the multi-species capability of our code, we study the bump-on-tail instability. This instability arises when a high-velocity electron beam creates a positive slope in the electron velocity distribution. We initialize the plasma with a bulk Maxwellian electron population and a tenuous, high-velocity beam that produces a pronounced bump in the tail of the distribution, and we track the evolution of the phase-space density and the associated electric field. During the linear growth phase, resonant electrons exchange energy with Langmuir waves, leading to exponential amplification of the electric field. As the instability evolves, the initially smooth electron distribution develops coherent phase-space structures, illustrating the code’s ability to capture nonlinear wave–particle interactions (\autoref{fig:bump-on-tail}).

![Simulation of the bump-on-tail instability. The numbers of pseudo-particles in the bulk and beam populations are equal, with a beam-to-bulk weight ratio of $3\times10^{-2}$. (a) Time evolution of the electric field energy. (b) Snapshot of phase space at $80\,\omega_{pe}^{-1}$. \label{fig:bump-on-tail}](figs/bump-on-tail.png)


Finally, we evaluated the computational performance of our implementation by comparing CPU and GPU runtimes on an AMD EPYC 7763 CPU and an NVIDIA A100 GPU, analyzing how the total runtime scales with the number of pseudoparticles. As a representative benchmark, we simulated ten drift velocities drawn from the two-stream dispersion relation (\autoref{fig:Run_time}). The results confirm the strong advantage of GPU acceleration: for the same workload, the GPU executes the simulation roughly two orders of magnitude faster than the CPU. In particular, for a system of 64,000 pseudoparticles, the GPU completes the full drift-scan in about six seconds after the initial compilation. Our benchmarks also indicate that GPU results depend on floating-point precision: running in 32-bit mode by manually disabling JAX’s x64 option reduces memory usage and improves speed, but can introduce deviations when compared to 64-bit results. Some of these differences also depend on details of the fitting range used to extract growth rates. For high-accuracy studies, we therefore recommend using the default 64-bit mode, with 32-bit remaining a useful option for rapid exploratory runs.

![(a) Comparison of total runtime between CPU and GPU. (b) Influence of pseudoparticle number on the two-stream instability sampling results. Growth rates extracted from exponential fits.\label{fig:Run_time}](figs/Run_time.png)

These results demonstrate that a fully JAX-based PIC code can deliver both high physical fidelity and excellent computational efficiency, making it suitable for rapid exploratory studies as well as larger-scale plasma simulations.

# Acknowledgement

This work was supported by the National Science Foundation under Grant No. PHY-2409066.
This work used the Jetstream2 at Indiana University through allocation PHY240054 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services \& Support (ACCESS) program which is supported by National Science Foundation grants \#213859, \#2138286, \#2138307, \#2137603 and \#2138296. This research used resources of the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231 using NERSC award NERSC DDR-ERCAP0030134. Aaron Tran was supported by the DOE Fusion Energy Sciences Postdoctoral Research Program, administered by the Oak Ridge Institute for Science and Education (ORISE) and Oak Ridge Associated Universities (ORAU) under DOE contract DE-SC0014664.

# References

