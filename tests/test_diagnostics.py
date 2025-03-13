import pytest
import jax.numpy as jnp
from jaxincell._diagnostics import diagnostics
from jaxincell._constants import epsilon_0, mu_0

def test_diagnostics():
    output = {
        'electric_field': jnp.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]),
        'external_electric_field': jnp.array([[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], [[0.0, 0.0, 0.5], [0.5, 0.5, 0.5]]]),
        'magnetic_field': jnp.array([[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], [[0.0, 0.0, 0.1], [0.1, 0.1, 0.1]]]),
        'external_magnetic_field': jnp.array([[[0.05, 0.0, 0.0], [0.0, 0.05, 0.0]], [[0.0, 0.0, 0.05], [0.05, 0.05, 0.05]]]),
        'velocity_electrons': jnp.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]),
        'velocity_ions': jnp.array([[[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], [[0.8, 0.9, 1.0], [1.1, 1.2, 1.3]]]),
        'grid': jnp.array([0.0, 1.0]),
        'dt': 0.1,
        'total_steps': 2,
        'plasma_frequency': 1.0,
        'dx': 0.1,
        'weight': 1.0,
        'mass_electrons': jnp.array([1.0]),
        'mass_ions': jnp.array([1.0])
    }

    diagnostics(output)

    assert 'electric_field_energy_density' in output
    assert 'electric_field_energy' in output
    assert 'magnetic_field_energy_density' in output
    assert 'magnetic_field_energy' in output
    assert 'dominant_frequency' in output
    assert 'plasma_frequency' in output
    assert 'kinetic_energy' in output
    assert 'kinetic_energy_electrons' in output
    assert 'kinetic_energy_ions' in output
    assert 'external_electric_field_energy_density' in output
    assert 'external_electric_field_energy' in output
    assert 'external_magnetic_field_energy_density' in output
    assert 'external_magnetic_field_energy' in output
    assert 'total_energy' in output

    assert jnp.allclose(output['electric_field_energy_density'], (epsilon_0/2) * jnp.sum(output['electric_field']**2))
    assert jnp.allclose(output['electric_field_energy'], (epsilon_0/2) * 0.5 * (jnp.sum(output['electric_field'][0]**2) + jnp.sum(output['electric_field'][1]**2)) * output['dx'])
    assert jnp.allclose(output['magnetic_field_energy_density'], 1/(2*mu_0) * jnp.sum(output['magnetic_field']**2, axis=-1))
    assert jnp.allclose(output['magnetic_field_energy'], jnp.array([3978.87358184, 7957.74716369]))
    assert jnp.allclose(output['dominant_frequency'], 0.0)
    assert jnp.allclose(output['plasma_frequency'], 1.0)
    assert jnp.allclose(output['kinetic_energy'], (1/2) * output['mass_electrons'][0] * output['weight'] * jnp.sum(jnp.sum(output['velocity_electrons']**2, axis=-1), axis=-1) + (1/2) * output['mass_ions'][0] * output['weight'] * jnp.sum(jnp.sum(output['velocity_ions']**2, axis=-1), axis=-1))
    assert jnp.allclose(output['kinetic_energy_electrons'], (1/2) * output['mass_electrons'][0] * output['weight'] * jnp.sum(jnp.sum(output['velocity_electrons']**2, axis=-1), axis=-1))
    assert jnp.allclose(output['kinetic_energy_ions'], (1/2) * output['mass_ions'][0] * output['weight'] * jnp.sum(jnp.sum(output['velocity_ions']**2, axis=-1), axis=-1))
    assert jnp.allclose(output['external_electric_field_energy_density'][0], (epsilon_0/2) * jnp.sum(output['external_electric_field'][0]**2))
    assert jnp.allclose(output['external_electric_field_energy'], jnp.array([1.10677348e-13, 2.21354696e-13]))
    assert jnp.allclose(output['external_magnetic_field_energy_density'], jnp.array([[ 9947.18395461,  9947.18395461],[ 9947.18395461, 29841.55186383]]))
    assert jnp.allclose(output['external_magnetic_field_energy'], jnp.array([ 994.71839546, 1989.43679092]))
    assert jnp.allclose(output['total_energy'], output["electric_field_energy"] + output["external_electric_field_energy"] + output["magnetic_field_energy"] + output["external_magnetic_field_energy"] + output["kinetic_energy"])

if __name__ == "__main__":
    pytest.main()