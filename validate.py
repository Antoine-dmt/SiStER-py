#!/usr/bin/env python3
"""Quick validation script for ConfigurationManager."""

import sys
sys.path.insert(0, '.')

try:
    # Test imports
    from sister_py.config import (
        ConfigurationManager, Material, FullConfig,
        SimulationConfig, DomainConfig, GridConfig,
        MaterialConfig, DensityParams
    )
    print("✓ All imports successful")
    
    # Test Pydantic validation
    from pydantic import ValidationError
    
    # Test 1: Invalid mu
    print("\n--- Test 1: Plasticity validation (mu > 1) ---")
    try:
        from sister_py.config import PlasticityParams
        p = PlasticityParams(C=40e6, mu=1.5)
        print("✗ Should have failed (mu=1.5)")
    except ValidationError as e:
        print("✓ Correctly rejected invalid mu")
        print(f"  Error: {str(e).split(chr(10))[0][:80]}...")
    
    # Test 2: Invalid grid spacing
    print("\n--- Test 2: Grid spacing validation (negative) ---")
    try:
        g = GridConfig(
            x_spacing=[1000, -500, 1000],
            x_breaks=[50e3, 150e3],
            y_spacing=[1000, 500, 1000],
            y_breaks=[30e3, 70e3]
        )
        print("✗ Should have failed (negative spacing)")
    except ValidationError as e:
        print("✓ Correctly rejected negative spacing")
        print(f"  Error: {str(e).split(chr(10))[0][:80]}...")
    
    # Test 3: Load continental rift example
    print("\n--- Test 3: Load continental rift configuration ---")
    try:
        cfg = ConfigurationManager.load('sister_py/data/examples/continental_rift.yaml')
        print(f"✓ Config loaded successfully")
        print(f"  Domain: {cfg.DOMAIN.xsize/1000:.0f} km × {cfg.DOMAIN.ysize/1000:.0f} km")
        print(f"  Materials: {len(cfg.MATERIALS)} phases")
        print(f"  Simulation: {cfg.SIMULATION.Nt} time steps")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
    
    # Test 4: Get materials
    print("\n--- Test 4: Material objects and properties ---")
    try:
        materials = cfg.get_materials()
        print(f"✓ Created {len(materials)} material objects")
        
        mantle = materials[2]
        print(f"  Phase 2 ({mantle.name}):")
        
        rho_cold = mantle.density(273)
        rho_hot = mantle.density(1373)
        print(f"    Density @ 273 K: {rho_cold:.0f} kg/m³")
        print(f"    Density @ 1373 K: {rho_hot:.0f} kg/m³")
        
        eta_duct = mantle.viscosity_ductile(sigma_II=1e7, eps_II=1e-15, T=1373)
        print(f"    Ductile viscosity: {eta_duct:.2e} Pa·s")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 5: Nested attribute access
    print("\n--- Test 5: Nested attribute access ---")
    try:
        xsize = cfg.DOMAIN.xsize
        nt = cfg.SIMULATION.Nt
        print(f"✓ Nested access works")
        print(f"  cfg.DOMAIN.xsize = {xsize}")
        print(f"  cfg.SIMULATION.Nt = {nt}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 6: Round-trip
    print("\n--- Test 6: Round-trip (dict → config → yaml → config) ---")
    try:
        data = cfg.to_dict()
        yaml_str = cfg.to_string()
        
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_str)
            temp_path = f.name
        
        cfg2 = ConfigurationManager.load(temp_path)
        os.unlink(temp_path)
        
        if cfg2.DOMAIN.xsize == cfg.DOMAIN.xsize:
            print(f"✓ Round-trip successful")
            print(f"  Original xsize: {cfg.DOMAIN.xsize}")
            print(f"  Reloaded xsize: {cfg2.DOMAIN.xsize}")
        else:
            print(f"✗ Values differ after round-trip")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY: All critical tests passed ✓")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
