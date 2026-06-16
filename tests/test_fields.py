def test_E_from_Gauss_1D_FFT_zero_mode_and_shape():
    """Test jaxincell._fields.E_from_Gauss_1D_FFT.

    Cases to implement:
    - zero charge density returns a finite zero electric field with matching shape.
    - a balanced sinusoidal charge density produces finite values with no NaNs.
    - the k=0 protection branch is exercised without leaking a DC field.
    """


def test_E_from_Poisson_1D_FFT_zero_average_and_sinusoid():
    """Test jaxincell._fields.E_from_Poisson_1D_FFT.

    Cases to implement:
    - zero charge density returns a finite zero electric field.
    - a single Fourier-mode charge density gives the expected phase-shifted field.
    - the phi_k[0] zero-average branch removes the potential DC mode.
    """


def test_E_from_Gauss_1D_Cartesian_solves_discrete_divergence():
    """Test jaxincell._fields.E_from_Gauss_1D_Cartesian.

    Cases to implement:
    - a small hand-computable charge-density vector solves the lower-triangular divergence system.
    - output has the same grid length as the input.
    - values remain finite for balanced positive and negative charges.
    """


def test_curlE_boundary_condition_cases():
    """Test jaxincell._fields.curlE.

    Cases to implement:
    - periodic, reflective, and absorbing field boundary conditions use the expected ghost cells.
    - constant E_y/E_z fields produce zero curl.
    - linear E_y/E_z fields produce the expected finite-difference signs.
    """


def test_curlB_boundary_condition_cases():
    """Test jaxincell._fields.curlB.

    Cases to implement:
    - periodic, reflective, and absorbing field boundary conditions use the expected ghost cells.
    - the internal roll by -1 is captured with a hand-computable staggered-grid example.
    - constant B_y/B_z fields produce zero curl where the chosen boundary permits it.
    """


def test_field_update_variants_zero_sources_and_current_response():
    """Test jaxincell._fields.field_update, field_update1, and field_update2.

    Cases to implement:
    - zero E, B, and current leave both fields unchanged.
    - zero curl with nonzero current updates only E by the Ampere current term.
    - output shapes match the input field shapes for all update variants.
    """


def test_field_update_variants_call_low_level_curls_consistently():
    """Test orchestration in jaxincell._fields.field_update, field_update1, and field_update2.

    Cases to implement:
    - compare each update variant against explicit calls to curlE and curlB.
    - verify the sequencing difference between simultaneous update, E-first update, and B-first update.
    - cover nonperiodic boundary conditions through the full update path.
    """
