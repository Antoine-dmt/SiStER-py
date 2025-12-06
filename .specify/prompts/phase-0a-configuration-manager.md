---
phase: 0a
component: ConfigurationManager
description: YAML-based configuration system for SiSteR-py simulations
dependencies: []
estimated_time: 3-5 days
---

# Phase 0A: ConfigurationManager

## Project Context

SiSteR-py follows the SiSteR MATLAB paradigm where a **single input file** drives the entire simulation. We use YAML (not MATLAB .m) for accessibility and maintainability. The ConfigurationManager must load, validate, and provide access to all simulation parameters from a single YAML file.

This component is **CRITICAL** because:
- All other components (Grid, Material, Solver, etc.) depend on ConfigurationManager
- It enables easy package distribution (example configs included)
- Users copy example YAML and modify only what they need (no code changes)
- Reproducibility: every simulation's config exported with outputs
- Extensibility: new physics added to YAML schema without modifying code

## Design Principles (from Constitution)

### I. Single-File Input Paradigm (Binding)
- One YAML file drives entire execution (mirroring SiSteR MATLAB `SiStER_Input_File_*.m`)
- Users modify only config parameters; code remains unchanged
- All simulations must be reproducible via saved config files
- YAML format (not JSON) ensures human readability, comments, version control compatibility

### III. Performance-First (Numba-Ready Code)
- Config loading must complete in < 100 ms
- No nested Python objects in config data structures (Numba compatibility)
- All material property accessors vectorizable (return NumPy arrays, not lists)

### V. Test-First Implementation
- Unit tests written before implementation
- Round-trip testing: load → modify → save → reload → bit-identical
- Acceptance criteria from this prompt are binding
- Coverage target: > 90%

## Specification

Create **ConfigurationManager** class that:

1. **Loads YAML configuration files** (pyyaml library)
2. **Validates all parameters** against schema (Pydantic v2)
3. **Creates Material objects** from config MATERIALS section
4. **Creates boundary condition dict** from config BC section
5. **Creates grid config dict** from config GRID section
6. **Provides read-only access** to all simulation parameters (nested attributes)
7. **Exports config** to YAML file, dict, or formatted string for reproducibility

## Requirements

### Loading & Parsing
- Use `pyyaml` library to load YAML files
- Handle multi-line strings, lists, nested dicts
- Preserve comments in YAML (for round-trip export)
- Support environment variable substitution: `${HOME}/data/file.nc` → expands to user's home directory
- Support YAML includes: `!include "base_config.yaml"` loads another YAML file

### Validation via Pydantic v2

Create BaseModel classes for each section:

```python
class SimulationConfig(BaseModel):
    Nt: int  # Total time steps (must be > 0)
    dt_out: int  # Output interval (must be > 0)
    output_dir: str  # Output directory path

class DomainConfig(BaseModel):
    xsize: float  # Domain width in meters (must be > 0)
    ysize: float  # Domain height in meters (must be > 0)

class GridConfig(BaseModel):
    # Variable spacing: defined by zone-wise cell sizes
    x_spacing: list[float]  # Cell widths in each zone (all > 0)
    x_breaks: list[float]  # Zone boundaries in x (must be increasing)
    y_spacing: list[float]  # Cell heights in each zone (all > 0)
    y_breaks: list[float]  # Zone boundaries in y (must be increasing)

class DensityParams(BaseModel):
    rho0: float  # Reference density (kg/m³, must be > 0)
    alpha: float  # Temperature gradient (1/K, typically |α| < 0.1)

class DuctileCreepParams(BaseModel):
    A: float  # Pre-exponential factor [Pa^(-n)·s^(-1)], must be > 0
    E: float  # Activation energy (J/mol, must be ≥ 0)
    n: float  # Stress exponent (must be > 0, typically 1-5)

class RheologyConfig(BaseModel):
    type: str  # "ductile", "plastic", "elastic"
    diffusion: DuctileCreepParams | None
    dislocation: DuctileCreepParams | None

class PlasticityParams(BaseModel):
    C: float  # Cohesion (Pa, must be ≥ 0)
    mu: float  # Friction coefficient (must be 0 < μ < 1)

class ElasticityParams(BaseModel):
    G: float  # Shear modulus (Pa, must be > 0, typically 1e10-1e11)

class ThermalParams(BaseModel):
    k: float  # Thermal conductivity (W/m/K, typically 1-10)
    cp: float  # Heat capacity (J/kg/K, typically 1000-1500)

class MaterialConfig(BaseModel):
    phase: int  # Phase ID (1-indexed)
    name: str  # Human-readable name
    density: DensityParams
    rheology: RheologyConfig | None
    plasticity: PlasticityParams | None
    elasticity: ElasticityParams | None
    thermal: ThermalParams | None

class BCConfig(BaseModel):
    type: str  # "velocity", "stress", "periodic"
    vx: float | None  # Velocity in x (m/s)
    vy: float | None  # Velocity in y (m/s)
    sxx: float | None  # Stress component (Pa)
    sxy: float | None  # Stress component (Pa)

class PhysicsConfig(BaseModel):
    elasticity: bool  # Enable elasticity?
    plasticity: bool  # Enable plasticity?
    thermal: bool  # Enable thermal evolution?

class SolverConfig(BaseModel):
    Npicard_min: int  # Minimum Picard iterations (must be > 0)
    Npicard_max: int  # Maximum Picard iterations
    conv_tol: float  # Convergence tolerance (typically 1e-8 to 1e-10)
    switch_to_newton: int  # Iteration to switch to Newton (0 = stay Picard)

class FullConfig(BaseModel):
    SIMULATION: SimulationConfig
    DOMAIN: DomainConfig
    GRID: GridConfig
    MATERIALS: list[MaterialConfig]
    BC: dict[str, BCConfig]  # Keys: "top", "bot", "left", "right"
    PHYSICS: PhysicsConfig
    SOLVER: SolverConfig
```

### Error Messages (Granular & Helpful)

Error messages must show **WHICH parameter failed and WHY**, not generic "validation failed":

```
❌ "friction at MATERIALS[1].plasticity.mu = 1.5, expected 0 < μ < 1"
❌ "Grid spacing must be positive, got -500 at GRID.x_spacing[1]"
❌ "Activation energy must be ≥ 0, got -100 at MATERIALS[0].rheology.diffusion.E"
❌ "Grid zone boundaries not monotonic: x_breaks = [50e3, 30e3, 140e3]"
```

Collect **all validation errors** before raising (not just first one).

### Material Objects

Method `get_materials()` creates Material instances:

```python
def get_materials(self) -> dict[int, Material]:
    """Create Material objects from config MATERIALS section.
    
    Returns:
        dict mapping phase_id → Material instance
        Material has methods:
        - viscosity_ductile(sigma_II, eps_II, T) → η
        - viscosity_plastic(sigma_II, P) → η or ∞
        - viscosity_effective(...) → min(ductile, plastic)
        - density(T) → ρ(T)
    """
```

Material class is simple wrapper with no validation (validation already done on load).

### Export Methods

```python
def to_dict(self) -> dict:
    """Return full config as nested dict (JSON-serializable)."""

def to_yaml(self, filepath: str) -> None:
    """Save config to YAML file (round-trip safe, 6 sig figs precision)."""

def to_string(self) -> str:
    """Return formatted config as printable string (for stdout, logs)."""

def validate(self) -> None:
    """Re-validate current config (useful after programmatic modifications)."""
```

## Constraints

### YAML Schema (Fixed Structure)
Must match SiSteR MATLAB input file format exactly (see EXAMPLE below).

### Pydantic v2 (Required)
- Use Pydantic v2 (not v1)
- All validation errors collected and reported together
- Custom validators for cross-field checks (e.g., x_breaks monotonic, phase IDs unique)

### Unit System (SI Throughout)
- All parameters in SI units
- **Temperature**: Kelvin (NOT Celsius)
- **Stress/Pressure**: Pascals (Pa)
- **Viscosity**: Pa·s
- **Energy**: Joules per mole (J/mol)
- **Density**: kg/m³
- **Position/Distance**: meters (m)

### Performance
- Config load: < 100 ms for 1000-line YAML file
- Round-trip export: 6 significant figures precision (e.g., `1.23456e-18`)

### Package Structure
- Module location: `sister_py/config.py`
- Example configs: `sister_py/data/examples/*.yaml`
- Default config: `sister_py/data/defaults.yaml` (hardcoded sensible values for all params)

## Accessibility Requirements

### Example YAML Files
Include well-documented examples in package:

```yaml
# sister_py/data/examples/continental_rift.yaml

# Grid spacing: three zones with different resolutions
# Zone 1: 0 to 50 km at 2000 m spacing
# Zone 2: 50 to 140 km at 500 m spacing (fine resolution)
# Zone 3: 140 to 170 km at 2000 m spacing
x_spacing: [2000, 500, 2000]
x_breaks: [50e3, 140e3]
```

### Validation Messages Suggest Fixes
Instead of just rejecting, offer remediation:
- "friction coefficient out of range: 1.5 > 1.0, try 0.3-0.6"
- "Grid spacing must be > 0, got -500, check zone definitions"

### Quick-Start Guide
Include "5-Minute Quick Start" documentation:
1. Copy example YAML from `~/.sister_py/examples/`
2. Modify 3-4 parameters (domain size, viscosity, friction)
3. Run simulation
4. Config auto-exported with outputs

### Installation
`pip install sister-py` includes examples in `~/.sister_py/`

## Acceptance Criteria (Binding)

- [ ] Load `continental_rift.yaml` without errors (real MATLAB input converted)
- [ ] Reject invalid config with **granular** error messages:
  - μ=1.5 → "friction > 1.0, expected 0 < μ < 1"
  - viscosity_max < viscosity_min → "bounds reversed"
  - Grid spacing negative → list which zone
  - x_breaks not monotonic → list values
- [ ] Round-trip: `load() → modify param → save() → load() → bit-identical config`
- [ ] Performance: load 1000-line config in < 100 ms
- [ ] Export: `config.to_yaml(file)` maintains 6 significant figures
- [ ] Materials: `config.get_materials()` returns dict of Material objects
- [ ] All Pydantic validators working (tests verify custom checks)
- [ ] Comments in YAML preserved after round-trip
- [ ] Environment variables expanded: `${HOME}/data/` → actual path
- [ ] Nested attribute access works: `cfg.DOMAIN.xsize`, `cfg.MATERIALS[0].density.rho0`

## Example YAML Configuration

```yaml
SIMULATION:
  Nt: 1600
  dt_out: 20
  output_dir: "./results"

DOMAIN:
  xsize: 170e3  # 170 km
  ysize: 60e3   # 60 km

GRID:
  x_spacing: [2000, 500, 2000]  # m
  x_breaks: [50e3, 140e3]        # m (zone boundaries)
  y_spacing: [2000, 500, 2000]
  y_breaks: [7e3, 40e3]

MATERIALS:
  - phase: 1
    name: "Sticky Layer"
    density:
      rho0: 1000      # kg/m³
      alpha: 0        # 1/K
    rheology:
      type: "ductile"
      diffusion:
        A: 0.5e-18    # Pa^(-n)·s^(-1)
        E: 0          # J/mol
        n: 1          # stress exponent
      dislocation:
        A: 0.5e-18
        E: 0
        n: 1
    plasticity:
      C: 40e6         # Pa
      mu: 0.6         # friction coefficient
    elasticity:
      G: 1e18         # Pa
    thermal:
      k: 3            # W/m/K
      cp: 1000        # J/kg/K

  - phase: 2
    name: "Mantle"
    density:
      rho0: 3300
      alpha: 3e-5
    rheology:
      type: "ductile"
      diffusion:
        A: 1e-21
        E: 400e3
        n: 3.5
      dislocation:
        A: 1.9e-16
        E: 540e3
        n: 3.5
    plasticity:
      C: 0
      mu: 0.6
    elasticity:
      G: 6.4e10
    thermal:
      k: 4.5
      cp: 1250

BC:
  top:
    type: "velocity"
    vx: 1e-10   # m/s
    vy: 0
  bot:
    type: "velocity"
    vx: 0
    vy: 0
  left:
    type: "velocity"
    vx: 1e-10
    vy: 0
  right:
    type: "velocity"
    vx: 1e-10
    vy: 0

PHYSICS:
  elasticity: true
  plasticity: true
  thermal: true

SOLVER:
  Npicard_min: 10
  Npicard_max: 100
  conv_tol: 1e-9
  switch_to_newton: 0
```

## Example Usage (Tests Must Cover)

```python
from sister_py.config import ConfigurationManager

# Load config from YAML file
cfg = ConfigurationManager.load("continental_rift.yaml")

# Access nested parameters (attribute access)
print(cfg.DOMAIN.xsize)  # 170000.0
print(cfg.MATERIALS[0].density.rho0)  # 1000.0

# Get Material objects ready for simulation
materials = cfg.get_materials()
print(materials[1].phase)  # 2

# Compute material viscosity
eta = materials[1].viscosity_ductile(
    sigma_II=1e7,       # Pa
    eps_II=1e-15,       # /s
    T=1200              # K
)

# Modify parameters programmatically
cfg.SIMULATION.Nt = 100
cfg.validate()  # Re-validate after changes

# Export for reproducibility
cfg.to_yaml("my_run.yaml")
print(cfg.to_string())  # Print to stdout
```

## Testing Strategy

### Unit Tests (Pydantic validation)
- Valid YAML loads without error
- Invalid parameters rejected with granular messages
- Custom validators working (zone boundaries, friction range, etc.)

### Round-Trip Tests
- Load YAML → modify → save → load → compare with original
- Verify 6 sig figs precision maintained

### Performance Tests
- 1000-line config loads in < 100 ms
- 10,000 sequential material property accesses in < 100 ms

### Integration Tests
- ConfigurationManager passed to Grid, Material, Solver
- All downstream components accept config objects correctly

## Dependencies

- `pyyaml>=6.0` – YAML parsing
- `pydantic>=2.0` – Validation
- `python>=3.10` – f-strings, type hints

## Deliverables

1. **sister_py/config.py**
   - ConfigurationManager class
   - All Pydantic BaseModel classes
   - Material wrapper class (simple, no logic)

2. **tests/test_config.py**
   - Validation tests (valid/invalid YAML)
   - Round-trip tests
   - Performance tests
   - Integration tests with mock Grid/Material

3. **sister_py/data/examples/*.yaml**
   - `continental_rift.yaml` (continental rifting)
   - `subduction.yaml` (subduction zone)
   - `shear_flow.yaml` (simple shear)

4. **sister_py/data/defaults.yaml**
   - Sensible defaults for all parameters

5. **Documentation**
   - API reference (docstrings in code)
   - "5-Minute Quick Start" guide
   - YAML schema documentation

## Success Criteria

✅ All acceptance criteria met
✅ Round-trip export produces bit-identical YAML
✅ Error messages guide users to valid parameter ranges
✅ Config load < 100 ms, material access < 100 ms
✅ Tests cover all acceptance criteria
✅ Example configs included and documented
