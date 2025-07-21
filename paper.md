---
title: 'Gala: A Python package for galactic dynamics'
tags:
  - Python
  - plasma
  - dynamics
authors:
  - name: Adrian M. Price-Whelan
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, United States
   index: 1
   ror: 00hx57361
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Particle‐in‐cell (PIC) methods are widely used to study the dynamics of charged particles interacting with electromagnetic fields. In a typical PIC framework, the domain is discretized on a grid and particles are represented by pseudo particles, with evolution carried out via a field solver, a particle pusher, and suitable boundary conditions. Jaxincell implements 1D3V PIC in JAX, enabling GPU acceleration, just in time (JIT) compilation, vectorized operations, and automatic differentiation to speed up simulations.


# Statement of need

Accurate and efficient tools are essential for modeling plasmas, whether for rapid testing of analytical ideas or large-scale optimization tasks. There is a need for modern PIC software that combines speed, flexibility, and ease of use. Jaxincell meets this need, offering high-performance simulations suitable for both research and educational purposes.

While PIC methods are well established, many codes remain closed‑source, written in legacy languages like Fortran, and carry high computational and maintenance costs. In contrast, Jaxincell is implemented entirely in Python, making it immediately accessible to a broad community of researchers and students. Furthermore, although the classic Boris push is both simple and robust, long‐term simulations can accumulate energy errors. To address this, Jaxincell includes not only the standard Boris algorithm but also an implicit, discretely energy‐conserving scheme.

# Structure

The core of our PIC code is the Vlasov–Maxwell system.  In particular, we solve

\begin{equation}
\label{pusher}
\partial_t f_s
+ v \cdot \nabla f_s
+ \frac{q_s}{m_s} \left( E + v \times B \right) \cdot \nabla_{\mathbf{u}} f_s = 0,
\end{equation}

\begin{equation}
\label{half_B}
\frac{\partial \mathbf{B}}{\partial t} 
+ \nabla \times \mathbf{E} = 0,
\end{equation}

\begin{equation}
\label{half_E}
\varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}
- c^2 \nabla \times \mathbf{B}
+ \mathbf{j} = 0.
\end{equation}

where $f_s(x,u)\approx\sum_{p\in s}w_p \delta(x-x_p)\delta(u-u_p)$ the distribution function is discretized by pseudo‑particle, $x$ is the position, $v$ is the velocity, $u=v\gamma$ is the proper velocity and $\gamma=\sqrt{1+u^2/c^2}$ is the Lorentz factor with c speed of light.

For notation, we will use s to label the species of particle, i to label the cell, n to label the step of time, and p to label the pseudo-particles. To reduce numerical noise, we represent each pseudo‑particle with a B‑spline shape function that spans three cells.  Its contribution to the charge density at cell $i$ is

\begin{equation}
\rho(x_p) =
\begin{cases}
\displaystyle
\frac{q}{\Delta x}
\left(
\frac{3}{4}
- \frac{(x_p - x_i)^2}{\Delta x^2}
\right),
& \text{if } \left| x_p - x_i \right| \le \frac{\Delta x}{2}, \\[10pt]
\displaystyle
\frac{q}{2 \Delta x}
\left(
\frac{3}{2}
- \frac{\left| x_p - x_i \right|}{\Delta x}
\right)^2,
& \text{if } \frac{\Delta x}{2} < \left| x_p - x_i \right| \le \frac{3 \Delta x}{2}, \\[10pt]
0, & \text{if } \left| x_p - x_i \right| > \frac{3 \Delta x}{2}.
\end{cases}
\end{equation}

Then, interpolation for fields is required as pseudo-particle weights span more than a single cell. Note, it is shifted by one cell due to ghost cells.

\begin{equation}
F(x_p)=
\frac{1}{2} F_{i-1} 
(
\frac{1}{2} + \frac{x_i - x_p}{\Delta x}
)^2
+
F_i 
(
\frac{3}{4} - \frac{(x_i - x_p)^2}{\Delta x^2}
)
+
\frac{1}{2} F_{i+1}
(
\frac{1}{2} - \frac{x_i - x_p}{\Delta x}
)^2.
\label{field}
\end{equation}

 We implement two standard methods, one explicit and one implicit. For explict method, we use the Boris Algorithm that have the following steps. 

1: Initialization with $E_i^n,B_i^n,v_p^n,x_p^{n-\frac{1}{2}},x_p^n,x_p^{n+\frac{1}{2}}$

2: Prepare the field $E(x_p^n),B(x_p^n)$ for the particle pusher by \autoref{field}

3: Push the particle as follows from \autoref{pusher}. 

(a) Electric half‑kick: $$v_p^{n+1/2}=v_p^n+\frac{q_p}{2m_p{\Delta t}} E(x_p^n) ,$$

(b) Magnetic rotation with second electric half‑kick: 

$${v_p^{n+1}} = \text{BorisRotate}(v_p^{n+1/2}, B(x_p^n))+\frac{q_p}{2m_p{\Delta t}} E(x_p^n),$$


(c) Position update (with centered interpolation): 

$$x_p^{n+\frac{3}{2}} = x_p^{n+\frac{1}{2}}+v_p^{n+1}{\Delta t}, x_p^{n+1}=\frac{1}{2}(x_p^{n+\frac{3}{2}}+x_p^{n+\frac{1}{2}})$$

4: Update the field according to \autoref{half_E} and \autoref{half_B}. 

(a) Electric half‑step:    

$$E_i^{n+\frac{1}{2}}= E_i^n +(c^2\nabla_i\times B_i^n - \frac{j_i^n}{\epsilon_0})\frac{\Delta t}{2}$$

(b) Magnetic full‑step:

$$B_i^{n+1} = B_i^n - (\nabla_i\times E_i^{n+\frac{1}{2}})\Delta t$$

(c) Electric half‑step:

$$ E_i^{n+1} = E_i^{n+\frac{1}{2}} +(c^2\nabla_i\times B_i^{n+1} - \frac{j_i^{n+1}}{\epsilon_0})\frac{\Delta t}{2}. $$

5: Save the carry for next step $E_i^{n+1},B_i^{n+1},v_p^{n+1},x_p^{n+\frac{1}{2}},x_p^{n+1},x_p^{n+\frac{3}{2}}$

Implicit method is similar but run iteration to solve the exact system of equation through Picard iteration. For simplicity we will set magnetic field to 0 and the Crank–Nicolson step follows:

1: Initialization with $E_i^n,v_p^n,x_p^n$

2: Picard iteration with intial guess $x_p^{n+\frac{1}{2}}=x_p^n$


(a) Prepare the field $E(x_p^{n+\frac{1}{2}})$ for the particle pusher

(b) Push the particle as follows from \autoref{pusher}: 

$$v_p^{n+1}=v_p^n+\frac{q_p}{m_p{\Delta t}} E(x_p^{n+\frac{1}{2}}), v_p^{n+\frac{1}{2}}=\frac{1}{2}(v_p^{n}+v_p^{n+1}),$$ 

$$x_p^{n+1} = x_p^n+v_p^{n+1/2}\Delta t$$

(c) Update the field according to \autoref{half_E}

$$ E_i^{n+1} = E_i^n  - \frac{1}{\epsilon_0} j_i^{n+1} \Delta t.$$

3: Check convergence. If $|E_\text{new}-E_\text{old}|<\text{tol}$, save  $E_i^{n+1},v_p^{n+1},x_p^{n+1}$ for next step. Otherwise, set $x_p^{n+\frac{1}{2}} = \frac{1}{2}\bigl(x_p^n + x_p^{n+1}\bigr)$ and return to step 2

# Capabilities

Two-stream instability, Landau damping, and Weibel instability are used for testing the correctness of the algorithms.



# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References