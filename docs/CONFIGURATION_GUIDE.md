# SiSteR-py Configuration System

## Quick Start (5 Minutes)

### 1. Install SiSteR-py

```bash
pip install -e .
```

### 2. Load a Configuration

```python
from sister_py import ConfigurationManager

# Load configuration from YAML
cfg = ConfigurationManager.load('sister_py/data/examples/continental_rift.yaml')

# Access configuration sections
print(f"Domain size: {cfg.DOMAIN.xsize} m")
print(f"Simulation steps: {cfg.SIMULATION.Nt}")
```

### 3. Get Material Objects

```python
# Create material objects from configuration
materials = cfg.get_materials()

# Access specific material
mantle = materials[2]

# Compute material properties
density = mantle.density(T=1373)  # at 1100 °C
viscosity = mantle.viscosity_ductile(sigma_II=1e7, eps_II=1e-15, T=1373)

print(f"Mantle density at 1373 K: {density:.0f} kg/m³")
print(f"Mantle viscosity: {viscosity:.2e} Pa·s")
```

### 4. Modify and Save

```python
# Export to dictionary (for modification)
data = cfg.to_dict()
data['DOMAIN']['xsize'] = 200e3  # Change domain width

# Re-create config
cfg_modified = ConfigurationManager.load(data)

# Save to new file
cfg_modified.to_yaml('config_modified.yaml')
```

---

## Configuration File Structure

### SIMULATION

Defines time stepping:
- `Nt` (int): Total number of time steps (> 0)
- `dt_out` (int): Output frequency in steps (> 0)
- `output_dir` (str): Directory for output files

```yaml
SIMULATION:
  Nt: 1600
  dt_out: 20
  output_dir: "./results"
```

### DOMAIN

Defines computational domain:
- `xsize` (float): Domain width in x-direction (m, > 0)
- `ysize` (float): Domain height in y-direction (m, > 0)

```yaml
DOMAIN:
  xsize: 170e3  # 170 km
  ysize: 60e3   # 60 km
```

### GRID

Defines grid spacing and zone boundaries:
- `x_spacing` (list): Grid cell size per zone (m)
- `x_breaks` (list): Zone boundary positions in x (m)
- `y_spacing` (list): Grid cell size per zone (m)
- `y_breaks` (list): Zone boundary positions in y (m)

Example with 3 zones: coarse [50-140 km] → fine [140-150 km] → coarse [150-170 km]

```yaml
GRID:
  x_spacing: [2000, 500, 2000]      # Cell sizes: 2000 m, 500 m, 2000 m
  x_breaks: [50e3, 140e3]           # Boundaries at 50 km, 140 km
  y_spacing: [2000, 500, 2000]
  y_breaks: [7e3, 40e3]
```

### MATERIALS

List of material phases:
- `phase` (int): Unique phase ID (> 0)
- `name` (str): Material name
- `density`: Density parameters (required)
- `rheology`: Ductile rheology (optional)
- `plasticity`: Mohr-Coulomb plasticity (optional)
- `elasticity`: Elastic parameters (optional)
- `thermal`: Thermal properties (optional)

#### Density Parameters
- `rho0` (float): Reference density (kg/m³, > 0)
- `alpha` (float): Thermal expansion coefficient (1/K)

Model: ρ(T) = ρ₀ · (1 - α · ΔT)

Typical ranges:
- Oceanic crust: ρ₀ ≈ 2800 kg/m³
- Mantle: ρ₀ ≈ 3300 kg/m³
- Alpha: 1–4 × 10⁻⁵ K⁻¹

#### Rheology Parameters (Power-Law Creep)

Both diffusion and dislocation creep follow: ε̇ = A·σⁿ·exp(-E/RT)

- `A` (float): Creep constant (Pa⁻ⁿ·s⁻¹, > 0)
- `E` (float): Activation energy (J/mol, ≥ 0)
- `n` (float): Stress exponent (> 0, typically 1–5)

Typical ranges:
| Mechanism | A | E | n |
|-----------|---|---|---|
| Diffusion creep | 1e-21 | 0–300 kJ/mol | 1.0 |
| Dislocation creep (upper crust) | 1e-16 | 500 kJ/mol | 2.5–3.5 |
| Dislocation creep (mantle) | 1.9e-16 | 540 kJ/mol | 3.5 |

#### Plasticity Parameters (Mohr-Coulomb)

Model: σ_Y = (C + μ·P)·cos(arctan(μ))

- `C` (float): Cohesion (Pa, ≥ 0)
- `mu` (float): Friction coefficient (0 < μ < 1)

Typical ranges:
- Cohesion: 0–100 MPa
- Friction: 0.3–0.6

#### Elasticity Parameters
- `G` (float): Shear modulus (Pa, > 0)

Typical: 5×10¹⁰ – 7×10¹⁰ Pa

#### Thermal Parameters
- `k` (float): Thermal conductivity (W/m/K)
- `cp` (float): Specific heat capacity (J/kg/K)

Typical:
- Conductivity: 2.5–4.5 W/m/K
- Specific heat: 1000–1250 J/kg/K

### BC (Boundary Conditions)

Dictionary of boundary conditions (keys: 'top', 'bottom', 'left', 'right'):
- `type` (str): 'velocity' or 'stress'
- `vx`, `vy` (float): Velocity components (m/s, for velocity BC)
- `sxx`, `sxy` (float): Stress components (Pa, for stress BC)

```yaml
BC:
  top:
    type: "velocity"
    vx: 1e-10
    vy: 0
  bottom:
    type: "velocity"
    vx: 0
    vy: 0
```

### PHYSICS

Physics flags:
- `elasticity` (bool): Enable elastic deformation
- `plasticity` (bool): Enable plastic yield
- `thermal` (bool): Enable thermal evolution

```yaml
PHYSICS:
  elasticity: true
  plasticity: true
  thermal: true
```

### SOLVER

Nonlinear solver parameters:
- `Npicard_min` (int): Minimum Picard iterations (> 0)
- `Npicard_max` (int): Maximum Picard iterations (> 0)
- `conv_tol` (float): Convergence tolerance (> 0)
- `switch_to_newton` (int): Iteration to switch to Newton (≥ 0)

```yaml
SOLVER:
  Npicard_min: 10
  Npicard_max: 100
  conv_tol: 1e-9
  switch_to_newton: 0
```

---

## API Reference

### ConfigurationManager

Load and manage simulation configurations.

#### Methods

##### `ConfigurationManager.load(filepath: str) → ConfigurationManager`

Load and validate YAML configuration.

**Args:**
- `filepath`: Path to YAML file (supports `${HOME}`, `$USER` expansion)

**Returns:** ConfigurationManager instance

**Raises:**
- `FileNotFoundError`: File does not exist
- `ValidationError`: Configuration invalid

**Example:**
```python
cfg = ConfigurationManager.load('config.yaml')
```

##### `get_materials() → Dict[int, Material]`

Create Material objects from MATERIALS config.

**Returns:** Dictionary mapping phase_id → Material

**Example:**
```python
materials = cfg.get_materials()
mantle = materials[2]
```

##### `to_dict() → Dict[str, Any]`

Export configuration as nested dictionary.

**Returns:** JSON-serializable dict

##### `to_yaml(filepath: str) → None`

Save configuration to YAML file.

**Args:**
- `filepath`: Output path

##### `to_string() → str`

Export configuration as YAML string.

**Returns:** Formatted YAML

##### `validate() → None`

Re-validate current configuration.

**Raises:** ValidationError if invalid

---

### Material

Material properties and rheology calculations.

#### Properties

##### `phase: int`
Phase ID

##### `name: str`
Material name

#### Methods

##### `density(T: float) → float`

Thermal expansion model: ρ(T) = ρ₀ · (1 - α · ΔT)

**Args:**
- `T`: Temperature (K)

**Returns:** Density (kg/m³)

##### `viscosity_ductile(sigma_II: float, eps_II: float, T: float) → float`

Power-law creep viscosity.

**Args:**
- `sigma_II`: Second invariant of deviatoric stress (Pa)
- `eps_II`: Second invariant of strain rate (1/s)
- `T`: Temperature (K)

**Returns:** Viscosity (Pa·s)

##### `viscosity_plastic(sigma_II: float, P: float) → float`

Mohr-Coulomb plastic viscosity.

**Args:**
- `sigma_II`: Second invariant of stress (Pa)
- `P`: Pressure (Pa)

**Returns:** Viscosity (Pa·s) or inf if not yielding

##### `viscosity_effective(sigma_II: float, eps_II: float, T: float, P: float) → float`

Effective viscosity (minimum of ductile and plastic).

**Args:**
- `sigma_II`: Second invariant of stress (Pa)
- `eps_II`: Second invariant of strain rate (1/s)
- `T`: Temperature (K)
- `P`: Pressure (Pa)

**Returns:** Effective viscosity (Pa·s)

---

## Examples

### Continental Rift

Located at: `sister_py/data/examples/continental_rift.yaml`

2-phase model: Sticky layer + mantle. Used for extensional tectonics.

```python
cfg = ConfigurationManager.load('sister_py/data/examples/continental_rift.yaml')
sticky_layer = cfg.get_materials()[1]
```

### Subduction Zone

Located at: `sister_py/data/examples/subduction.yaml`

2-phase model: Oceanic crust + mantle wedge. Used for convergent plate margins.

### Simple Shear

Located at: `sister_py/data/examples/shear_flow.yaml`

Pure viscous shear flow. Good for benchmarking and validation.

---

## SI Units

All parameters use International System (SI) units:

| Quantity | Unit | Symbol |
|----------|------|--------|
| Length | meter | m |
| Mass | kilogram | kg |
| Time | second | s |
| Temperature | kelvin | K |
| Stress | pascal | Pa |
| Viscosity | pascal·second | Pa·s |
| Density | kilogram/meter³ | kg/m³ |
| Energy | joule | J |
| Heat | joule/kilogram/kelvin | J/kg/K |

---

## Troubleshooting

### ValidationError: "greater than 0"
One of the positive-only fields (e.g., `Nt`, `xsize`) has zero or negative value.

**Fix:** Ensure all GT constraints are met. See typical ranges in Configuration File Structure above.

### ValidationError: "must be unique"
Two materials have the same phase ID.

**Fix:** Ensure each material in MATERIALS list has a unique `phase` value.

### ValidationError: "must be strictly increasing"
Grid zone boundaries are not monotonically increasing.

**Fix:** Ensure `x_breaks` and `y_breaks` satisfy: `boundaries[0] < boundaries[1] < ...`

### FileNotFoundError
Configuration file path is wrong or file doesn't exist.

**Fix:** Check file path and use absolute paths or relative to working directory.

---

## References

- **Power-Law Creep**: Hirth & Kohlstedt (2003), "Rheology of the upper mantle and the mantle wedge: A view from the experimentalists"
- **Mohr-Coulomb Plasticity**: Byerlee (1978), "Friction of rocks"
- **Fully-Staggered Grids**: Duretz et al. (2013), "Discretization errors and free surface stability in the finite difference and marker-in-cell method"
