import pytest

from jaxincell._constants import (
    epsilon_0, mu_0, speed_of_light, elementary_charge,
    mass_electron, mass_proton
)

q_m_electron      = elementary_charge / mass_electron # Electron charge-to-mass ratio
q_m_proton        = elementary_charge / mass_proton   # Proton charge-to-mass ratio

def test_epsilon_0():
    assert epsilon_0 == 8.85418782e-12, "Incorrect value for epsilon_0"

def test_mu_0():
    assert mu_0 == 1.25663706e-7, "Incorrect value for mu_0"

def test_speed_of_light():
    assert speed_of_light == 2.99792458e8, "Incorrect value for speed_of_light"

def test_charge_electron():
    assert elementary_charge == 1.60217663e-19, "Incorrect value for charge_electron"

def test_mass_electron():
    assert mass_electron == 9.10938371e-31, "Incorrect value for mass_electron"

def test_mass_proton():
    assert mass_proton == 1.67262193e-27, "Incorrect value for mass_proton"

def test_q_m_electron():
    expected_value = elementary_charge / mass_electron
    assert q_m_electron == expected_value, "Incorrect value for q_m_electron"

def test_q_m_proton():
    expected_value = elementary_charge / mass_proton
    assert q_m_proton == expected_value, "Incorrect value for q_m_proton"

if __name__ == "__main__":
    pytest.main()