# Boundary conditions

The box has two walls at $x = \pm L/2$. Each wall carries one code for particles and
one for fields (`0` periodic, `1` reflective, `2` absorbing), set in the domain
parameters. This page gives the formulas; {doc}`../user_guide/boundaries` discusses
when to use which.

## Particles

For a particle that has left the box through the left wall ($x < -L/2$) or the right
wall ($x > L/2$):

| code | position | velocity | charge, $q/m$ |
|---|---|---|---|
| `0` periodic | $x \to x \pm L$ | unchanged | unchanged |
| `1` reflective | $x \to -L - x$ (left), $x \to L - x$ (right) | $v_x \to -v_x$ | unchanged |
| `2` absorbing | $x \to x_0 - 1.5\,\Delta x$ (left), $x \to x_{N_x-1} + 3\,\Delta x$ (right) | $\mathbf v \to 0$ | set to zero |

The $y$ and $z$ coordinates are always wrapped into $[-L_y/2, L_y/2]$ and
$[-L_z/2, L_z/2]$. The same map, without the velocity and charge changes, is applied
to the half-step positions used for the current deposit. An absorbed particle stays
in the arrays at its parking position with zero charge, so its shape function never
overlaps the grid again and it drops out of every deposit and of the kinetic energy.

## Charge deposit near a wall

The quadratic spline of a particle within $1.5\,\Delta x$ of a wall extends beyond
the last cell centre. The part that falls beyond the wall is handled per wall code:

| code | charge beyond the wall |
|---|---|
| `0` periodic | added to the first (respectively last) cell |
| `1` reflective | added to the last (respectively first) cell, that is folded back onto the boundary cell |
| `2` absorbing | dropped |

Only the fraction of the cloud within half a cell beyond the last centre is
redistributed, which is the whole cloud for particles inside the box when the deposit
is made on cell centres.

## Field ghost cells

The curl operators need one value beyond each end of the grid. With $F$ standing for
the array being differentiated and the other field available for the absorbing case:

| code | left ghost | right ghost |
|---|---|---|
| `0` periodic | $F_{N_x-1}$ | $F_0$ |
| `1` reflective | $F_0$ | $F_{N_x-1}$ |
| `2` absorbing | outgoing-wave combination, below | outgoing-wave combination, below |

For the electric field at the left wall the absorbing ghost is

```{math}
E^{g}_y = -2c\,B_{z,0} - E_{y,0}, \qquad E^{g}_z = 2c\,B_{y,0} - E_{z,0},
```

so that the average of the ghost and boundary values satisfies
$\tfrac12(E^{g}_y + E_{y,0}) = -c B_{z,0}$ and $\tfrac12(E^{g}_z + E_{z,0}) = c B_{y,0}$:
the transverse field at the wall is that of a plane wave travelling in the $-x$
direction, which leaves the box. At the right wall the ghosts are
$E^{g}_y = 3E_{y,N_x-1} - 2cB_{z,N_x-1}$ and $E^{g}_z = 3E_{z,N_x-1} + 2cB_{y,N_x-1}$,
and the magnetic ghosts are the corresponding expressions with $E$ and $B$
interchanged and $c$ replaced by $1/c$. These are first-order absorbing conditions of
the Mur type {cite}`mur1981`: exact for normal incidence in the continuum limit, with a
small reflection at the discrete level.

When fields are interpolated to particles, two ghost values are needed on the left
(because the electric field lives on cell faces) and one on the right. Periodic and
reflective walls copy the corresponding interior values; absorbing walls use zeros.

## Filter

The three-point filter shifts the array by $\pm s$ cells. Periodic walls roll the
array, reflective walls clamp the index to the boundary cell, absorbing walls use zero
outside the box.

## Implicit scheme

The implicit scheme's deposit and gather wrap indices with the modulus operator,
which is the periodic condition, whatever the field codes are. Particle codes are
applied as above.
