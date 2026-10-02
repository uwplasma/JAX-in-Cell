# Physical constants

The constants used by the code are plain Python floats in SI units, importable from
the package:

```python
from jaxincell import epsilon_0, mu_0, speed_of_light, elementary_charge, mass_electron, mass_proton, boltzmann_constant
```

| name | value | unit |
|---|---|---|
| `epsilon_0` | 8.85418782e-12 | F/m |
| `mu_0` | 1.25663706e-6 | H/m |
| `speed_of_light` | 2.99792458e8 | m/s |
| `elementary_charge` | 1.60217663e-19 | C |
| `mass_electron` | 9.10938371e-31 | kg |
| `mass_proton` | 1.67262193e-27 | kg |
| `boltzmann_constant` | 1.380649e-23 | J/K |

Species charges and masses are expressed as multiples of `elementary_charge`,
`mass_electron` and `mass_proton` in the input, see {doc}`../user_guide/species`.
