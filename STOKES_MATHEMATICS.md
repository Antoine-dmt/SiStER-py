# Stokes Equation Deep Dive: Mathematics & Finite Difference Implementation

## Table of Contents
1. Continuous Stokes Equations
2. Staggered Grid Discretization
3. Finite Difference Stencils
4. System Assembly in Matrix Form
5. Boundary Conditions
6. Non-linear Rheology Coupling
7. Numerical Stability & Scaling

---

## 1. Continuous Stokes Equations

### 1.1 Fundamental Equations

**Momentum Balance** (Newton's 2nd law for slow flow):
$$\nabla \cdot \boldsymbol{\sigma} + \rho \mathbf{g} = 0$$

**Mass Conservation** (Continuity):
$$\nabla \cdot \mathbf{v} = 0$$

**Constitutive Relation** (Linear viscous fluid):
$$\boldsymbol{\sigma} = 2\eta \mathbf{\dot{\varepsilon}} - p \mathbf{I}$$

Where:
- $\boldsymbol{\sigma}$ = total stress tensor
- $\eta$ = dynamic viscosity (Pa·s)
- $\mathbf{\dot{\varepsilon}}$ = strain rate tensor
- $p$ = pressure (scalar)
- $\rho$ = density (kg/m³)
- $\mathbf{g}$ = gravitational acceleration (m/s²)

### 1.2 Component Form (2D)

**X-Momentum**:
$$\frac{\partial \sigma_{xx}}{\partial x} + \frac{\partial \sigma_{xy}}{\partial y} + \rho g_x = 0$$

**Y-Momentum**:
$$\frac{\partial \sigma_{xy}}{\partial x} + \frac{\partial \sigma_{yy}}{\partial y} + \rho g_y = 0$$

**Continuity**:
$$\frac{\partial v_x}{\partial x} + \frac{\partial v_y}{\partial y} = 0$$

### 1.3 Stress in Terms of Velocity

Using the constitutive relation:
$$\sigma_{ij} = 2\eta \dot{\varepsilon}_{ij} - p \delta_{ij}$$

Where:
$$\dot{\varepsilon}_{xx} = \frac{\partial v_x}{\partial x}, \quad \dot{\varepsilon}_{yy} = \frac{\partial v_y}{\partial y}, \quad \dot{\varepsilon}_{xy} = \frac{1}{2}\left(\frac{\partial v_x}{\partial y} + \frac{\partial v_y}{\partial x}\right)$$

Therefore:
$$\sigma_{xx} = 2\eta \frac{\partial v_x}{\partial x} - p$$
$$\sigma_{yy} = 2\eta \frac{\partial v_y}{\partial y} - p$$
$$\sigma_{xy} = \eta\left(\frac{\partial v_x}{\partial y} + \frac{\partial v_y}{\partial x}\right)$$

### 1.4 Substituting into Momentum (Navier-Stokes for Stokes regime)

**X-Momentum** becomes:
$$\frac{\partial}{\partial x}\left(2\eta \frac{\partial v_x}{\partial x} - p\right) + \frac{\partial}{\partial y}\left(\eta\left(\frac{\partial v_x}{\partial y} + \frac{\partial v_y}{\partial x}\right)\right) + \rho g_x = 0$$

Expanding (assuming $\eta$ constant):
$$2\eta \frac{\partial^2 v_x}{\partial x^2} - \frac{\partial p}{\partial x} + \eta \frac{\partial^2 v_x}{\partial y^2} + \eta \frac{\partial^2 v_y}{\partial x \partial y} + \rho g_x = 0$$

Simplifying using continuity $\frac{\partial v_x}{\partial x} + \frac{\partial v_y}{\partial y} = 0$:
$$\eta \nabla^2 v_x - \frac{\partial p}{\partial x} + \rho g_x = 0$$

Similarly for Y:
$$\eta \nabla^2 v_y - \frac{\partial p}{\partial y} + \rho g_y = 0$$

Or in vector form:
$$\eta \nabla^2 \mathbf{v} - \nabla p + \rho \mathbf{g} = 0$$

---

## 2. Staggered Grid (MAC Grid) Discretization

### 2.1 Grid Layout

```
      j-1      j      j+1      (column)
              
y   i+1   +-------+-------+
          |       |       |
        x |   v_y |       |
    ∧     |       |       |
    |   i +---p---+---p---+
    |     | v_x σ_xy|
    |   i-1 +-------+-------+
    |
    x -->
    (row)
```

**Node Types**:
- **Normal nodes** (•, P-nodes): $(x_j, y_i)$ — pressure $p_{i,j}$
- **Shear/Velocity nodes** (◊, V-nodes):
  - $v_x$ at $(x_{j+1/2}, y_i)$ — horizontal velocity
  - $v_y$ at $(x_j, y_{i+1/2})$ — vertical velocity
  - $\sigma_{xy}$ at $(x_{j+1/2}, y_{i+1/2})$ — shear stress

### 2.2 Grid Coordinates

**X-axis** (Nx+1 points, Nx cells):
```
x = [0, dx_1, dx_1+dx_2, ..., xsize]
```

**Y-axis** (Ny+1 points, Ny cells):
```
y = [0, dy_1, dy_1+dy_2, ..., ysize]
```

**Half-points** (for staggered nodes):
```
x_{j+1/2} = (x_j + x_{j+1}) / 2
y_{i+1/2} = (y_i + y_{i+1}) / 2
```

---

## 3. Finite Difference Stencils

### 3.1 First Derivative (Central Difference)

**At location (i,j)**:
$$\frac{\partial f}{\partial x}\bigg|_{i,j} \approx \frac{f_{i,j+1/2} - f_{i,j-1/2}}{\Delta x_j}$$

Where $\Delta x_j = x_{j+1/2} - x_{j-1/2}$ is the grid spacing.

Similarly for $y$:
$$\frac{\partial f}{\partial y}\bigg|_{i,j} \approx \frac{f_{i+1/2,j} - f_{i-1/2,j}}{\Delta y_i}$$

### 3.2 Second Derivative (Central Difference)

$$\frac{\partial^2 f}{\partial x^2}\bigg|_{i,j} \approx \frac{f_{i,j+1} - 2f_{i,j} + f_{i,j-1}}{\left(\Delta x_j\right)^2}$$

Or more generally (for variable spacing):
$$\frac{\partial^2 f}{\partial x^2}\bigg|_{i,j} \approx 2\left(\frac{f_{i,j+1} - f_{i,j}}{\Delta x_{j,j+1}(\Delta x_{j,j+1} + \Delta x_{j-1,j})} - \frac{f_{i,j} - f_{i,j-1}}{\Delta x_{j-1,j}(\Delta x_{j,j+1} + \Delta x_{j-1,j})}\right)$$

---

## 4. Momentum Equations on Staggered Grid

### 4.1 X-Momentum at $v_x$-node $(i, j+1/2)$

Discretize:
$$\frac{\partial \sigma_{xx}}{\partial x} + \frac{\partial \sigma_{xy}}{\partial y} + \rho g_x = 0$$

**Term 1: $\frac{\partial \sigma_{xx}}{\partial x}$** (σ_xx evaluated at left/right normal nodes)
$$\frac{\partial \sigma_{xx}}{\partial x}\bigg|_{i,j+1/2} = \frac{\sigma_{xx,i,j+1} - \sigma_{xx,i,j}}{\Delta x_j}$$

**Term 2: $\frac{\partial \sigma_{xy}}{\partial y}$** (σ_xy at above/below shear nodes)
$$\frac{\partial \sigma_{xy}}{\partial y}\bigg|_{i,j+1/2} = \frac{\sigma_{xy,i+1/2,j+1/2} - \sigma_{xy,i-1/2,j+1/2}}{\Delta y_i}$$

**Assembled FD Equation**:
$$\frac{\sigma_{xx,i,j+1} - \sigma_{xx,i,j}}{\Delta x_j} + \frac{\sigma_{xy,i+1/2,j+1/2} - \sigma_{xy,i-1/2,j+1/2}}{\Delta y_i} + \rho_{i,j+1/2} g_x = 0$$

Now substitute constitutive relations:
$$\sigma_{xx} = 2\eta \frac{\partial v_x}{\partial x} - p$$
$$\sigma_{xy} = \eta\left(\frac{\partial v_x}{\partial y} + \frac{\partial v_y}{\partial x}\right)$$

This gives a linear system in unknowns: $\{v_x, v_y, p\}$.

### 4.2 Y-Momentum at $v_y$-node $(i+1/2, j)$

Similarly:
$$\frac{\partial \sigma_{xy}}{\partial x} + \frac{\partial \sigma_{yy}}{\partial y} + \rho g_y = 0$$

Becomes:
$$\frac{\sigma_{xy,i+1/2,j+1/2} - \sigma_{xy,i+1/2,j-1/2}}{\Delta x_j} + \frac{\sigma_{yy,i+1,j} - \sigma_{yy,i,j}}{\Delta y_i} + \rho_{i+1/2,j} g_y = 0$$

### 4.3 Continuity at P-node $(i,j)$

$$\frac{\partial v_x}{\partial x} + \frac{\partial v_y}{\partial y} = 0$$

Discretized:
$$\frac{v_{x,i,j+1/2} - v_{x,i,j-1/2}}{\Delta x_j} + \frac{v_{y,i+1/2,j} - v_{y,i-1/2,j}}{\Delta y_i} = 0$$

---

## 5. Complete System Matrix

### 5.1 Solution Vector Ordering

Stack all unknowns at all grid points in a single vector $\mathbf{S}$:

For each grid point, order as: $(p, v_x, v_y)$

```
S = [p(1,1), vx(1,1), vy(1,1),  p(1,2), vx(1,2), vy(1,2), ... ]^T
```

Linear index for point $(i,j)$ in a Ny × Nx grid:
$$\text{point\_index} = (j-1) \times Ny + i$$
$$\text{row\_p} = 3 \times \text{point\_index} - 2$$
$$\text{row\_vx} = 3 \times \text{point\_index} - 1$$
$$\text{row\_vy} = 3 \times \text{point\_index}$$

### 5.2 Assembly Loop Pseudocode

```python
L = sparse_matrix(3*Nx*Ny, 3*Nx*Ny)
R = zeros(3*Nx*Ny)

for j in 1:Nx:
    for i in 1:Ny:
        
        # Row indices for this point
        in_p = 3*((j-1)*Ny + i) - 3  # pressure equation
        in_vx = in_p + 1             # vx equation
        in_vy = in_p + 2             # vy equation
        
        # PRESSURE EQUATION (continuity)
        if (i, j) is interior:
            # ∂vx/∂x + ∂vy/∂y = 0
            L[in_p, in_vx @ (i,j+1/2)] += 1/dx_j
            L[in_p, in_vx @ (i,j-1/2)] += -1/dx_j
            L[in_p, in_vy @ (i+1/2,j)] += 1/dy_i
            L[in_p, in_vy @ (i-1/2,j)] += -1/dy_i
            R[in_p] = 0
        else:
            # Boundary condition for pressure
            ...
        
        # VX EQUATION (x-momentum)
        if (i, j+1/2) is interior:
            # ∂σxx/∂x + ∂σxy/∂y + ρgx = 0
            # Terms with pressure: p_{i,j+1}, p_{i,j}
            L[in_vx, in_p @ (i,j+1)] += -1/dx_j
            L[in_vx, in_p @ (i,j)] += 1/dx_j
            
            # Terms with ∂²vx/∂x² (from 2η∂²vx/∂x²)
            L[in_vx, in_vx @ (i,j+1)] += 2*eta_i_jp / (dx_j * dx_jp)
            L[in_vx, in_vx @ (i,j)] += -2*eta_i_jp / (dx_j * dx_jp) - 2*eta_i_jm / (dx_j * dx_jm)
            L[in_vx, in_vx @ (i,j-1)] += 2*eta_i_jm / (dx_j * dx_jm)
            
            # Terms with ∂²vx/∂y² (from η∂²vx/∂y²)
            L[in_vx, in_vx @ (i+1,j)] += eta_ip_j / (dy_i * dy_ip)
            L[in_vx, in_vx @ (i,j)] += -eta_ip_j / (dy_i * dy_ip) - eta_im_j / (dy_i * dy_im)
            L[in_vx, in_vx @ (i-1,j)] += eta_im_j / (dy_i * dy_im)
            
            # Terms with ∂²vy/∂x∂y (from η∂²vy/∂x∂y)
            L[in_vx, in_vy @ (i+1,j+1)] += eta_ip_jp / (4*dx_j*dy_i)
            L[in_vx, in_vy @ (i+1,j)] += -eta_ip_jm / (4*dx_j*dy_i)
            L[in_vx, in_vy @ (i,j+1)] += -eta_im_jp / (4*dx_j*dy_i)
            L[in_vx, in_vy @ (i,j)] += eta_im_jm / (4*dx_j*dy_i)
            
            R[in_vx] = -rho_i_jp * g_x
        else:
            # Boundary condition (velocity or stress)
            ...
        
        # VY EQUATION (y-momentum)
        if (i+1/2, j) is interior:
            # Similar to vx but with x↔y, xx↔yy
            ...
        else:
            ...
```

### 5.3 System Form

$$\underbrace{\begin{bmatrix} L_{pp} & L_{p,vx} & L_{p,vy} \\ L_{vx,p} & L_{vx,vx} & L_{vx,vy} \\ L_{vy,p} & L_{vy,vx} & L_{vy,vy} \end{bmatrix}}_\text{L} \begin{bmatrix} \mathbf{p} \\ \mathbf{v}_x \\ \mathbf{v}_y \end{bmatrix} = \begin{bmatrix} \mathbf{0} \\ \rho \mathbf{g}_x \\ \rho \mathbf{g}_y \end{bmatrix}$$

The matrix $L$ is symmetric (important!), sparse (< 5% dense for staggered grid).

---

## 6. Boundary Conditions

### 6.1 Dirichlet (Velocity Prescribed)

On boundary edge, specify $v_x$ or $v_y$:
$$v_x = u_{\text{bc}} \quad \text{or} \quad v_y = v_{\text{bc}}$$

**In matrix**: Replace momentum equation row with:
$$1 \cdot v_x = u_{\text{bc}}$$

Example (top boundary, y = 0):
```python
for j in 1:Nx:
    i = 1  # top row
    row = row_vx(i, j)
    L[row, :] = 0
    L[row, (row+0)] = 1.0  # coefficient for vx
    R[row] = u_top
```

### 6.2 Neumann (Stress/Traction Prescribed)

Stress components on boundary:
$$\sigma \cdot \mathbf{n} = \mathbf{t}_{\text{bc}}$$

Example (top boundary, $y=0$, normal $\mathbf{n} = -\hat{y}$):
$$\sigma_{xy} = t_{x,\text{bc}}, \quad \sigma_{yy} = -t_{y,\text{bc}}$$

From constitutive relations:
$$\sigma_{yy} = 2\eta \frac{\partial v_y}{\partial y} - p$$

This couples $p$, $v_y$, and derivatives. Often replaced with free-slip conditions.

### 6.3 Pressure Anchor

The system $L\mathbf{S} = R$ is singular (pressure determined up to constant). 
Fix one pressure value to remove nullspace:

```python
# At grid point (i, j) = (IP, JP)
row = row_p(IP, JP)
L[row, :] = 0
L[row, row] = 1.0
R[row] = p_anchor  # e.g., 0
```

---

## 7. Non-Linear Rheology Coupling

### 7.1 Viscosity Dependence

In each Picard iteration:
1. Current solution gives strain rate: $\dot{\varepsilon}_{ij} = \frac{\partial v_i}{\partial x_j}$
2. Compute second invariant: $\dot{\varepsilon}_{II} = \sqrt{\frac{1}{2}(\dot{\varepsilon}_{xx}^2 + \dot{\varepsilon}_{yy}^2 + 2\dot{\varepsilon}_{xy}^2)}$
3. Update viscosity: $\eta = \eta(\dot{\varepsilon}_{II}, T, \sigma)$ via creep law
4. Reassemble $L$ matrix
5. Solve new $L\mathbf{S} = R$
6. Repeat until convergence

### 7.2 Creep Law Example (Dislocation)

$$\dot{\varepsilon}_{II} = A \sigma_{II}^n \exp\left(-\frac{E}{nRT}\right)$$

Rearrange for stress:
$$\sigma_{II} = \left(\frac{\dot{\varepsilon}_{II}}{A \exp(-E/nRT)}\right)^{1/n}$$

Effective viscosity:
$$\eta = \frac{\sigma_{II}}{2\dot{\varepsilon}_{II}} = \frac{1}{2A^{1/n}} \dot{\varepsilon}_{II}^{(1-n)/n} \exp\left(\frac{E}{nRT}\right)$$

### 7.3 Plasticity (Stress Capped)

Yield strength:
$$\sigma_Y = (C + \mu P) \cos(\arctan(\mu))$$

If $\sigma_{II} > \sigma_Y$, cap at:
$$\eta_{\text{eff}} = \frac{\sigma_Y}{2\dot{\varepsilon}_{II}}$$

---

## 8. Numerical Stability & Scaling

### 8.1 Condition Number Problem

Momentum equation (example):
$$\frac{\partial^2 v_x}{\partial x^2} \sim 10^{-6} \text{ (small spatial variations)}$$
$$\frac{\partial p}{\partial x} \sim 10^5 \text{ (large pressure gradients)}$$

Both are $O(1)$ physically but vastly different in magnitude numerically.

**Solution**: Scale equations before assembly:
$$K_c = \frac{2\eta_{\min}}{dx_{\max} + dy_{\max}} \quad \text{(momentum scale)}$$
$$K_b = \frac{4\eta_{\min}}{(dx_{\max} + dy_{\max})^2} \quad \text{(continuity scale)}$$

Discretized momentum equation becomes:
$$\frac{K_c}{2\eta} \times (\text{momentum eqn})$$

Continuity equation becomes:
$$K_b \times (\text{continuity eqn})$$

This balances matrix coefficients to $O(1)$, improving solver stability.

### 8.2 Preconditioner

For iterative solvers (e.g., GMRes), use preconditioner to accelerate convergence.
Common choice: diagonal scaling (Jacobi).

---

## 9. Validation: Analytical Solutions

### 9.1 Simple Shear

**Setup**: Channel flow, no gravity, linear velocity BC
- Top: $v_x = V$, bottom: $v_x = 0$
- Sides: periodic or stress-free

**Analytical solution**:
$$v_x(y) = V \frac{y - y_{\min}}{y_{\max} - y_{\min}}$$
$$\frac{\partial v_x}{\partial y} = \frac{V}{H}$$
$$\sigma_{xy} = \eta \frac{V}{H}$$
$$p = \text{const}$$

**Test**: Verify FD solution matches to machine precision.

### 9.2 Stagnant Gravity Column

**Setup**: No flow, uniform viscosity, gravity
- $v_x = v_y = 0$
- $\frac{\partial p}{\partial y} = \rho g$

**Analytical solution**:
$$p(y) = p_0 + \rho g (y - y_0)$$

**Test**: Solve Stokes with pure gravity, verify pressure satisfies hydrostatic.

---

## 10. Code Implementation Checklist

When implementing FD stencil assembly:

- [ ] Correct stencil coefficients (factor-of-2 errors common!)
- [ ] Consistent indexing: (i,j) → linear row/col
- [ ] Pressure anchor: fix one P value
- [ ] Boundary conditions: replace equation rows on edges
- [ ] Scaling: apply Kc, Kb
- [ ] Sparsity: use COO format during assembly, convert to CSR for solve
- [ ] Symmetry: check L = L^T (optional, helps debugging)
- [ ] Matrix size: verify 3×(Nx×Ny) × 3×(Nx×Ny)
- [ ] RHS size: verify 3×(Nx×Ny)
- [ ] Solver choice: direct (small grids), iterative (large, preconditioned)

---

## Summary: Key Equations & Stencils

**Momentum (at v-node)**:
$$\nabla p = \eta \nabla^2 \mathbf{v} + \rho \mathbf{g}$$

**Continuity (at p-node)**:
$$\nabla \cdot \mathbf{v} = 0$$

**Strain rate (computed from v)**:
$$\dot{\varepsilon}_{II} = \sqrt{\frac{1}{2}\text{Tr}(\mathbf{\dot{\varepsilon}}^2)}$$

**Viscosity (from rheology)**:
$$\eta = \eta(\dot{\varepsilon}_{II}, T, \sigma)$$

**Matrix form**:
$$\begin{bmatrix} 0 & \nabla^T \\ \nabla & \eta^{-1}\mathbf{M} \end{bmatrix} \begin{bmatrix} \mathbf{p} \\ \mathbf{v} \end{bmatrix} = \begin{bmatrix} 0 \\ \mathbf{f} \end{bmatrix}$$

Where M is the Helmholtz operator, f incorporates gravity.

---

**References**:
- Gerya & Yuen (2003) on FD staggered grid Stokes
- Elman et al. on saddle-point systems
- SiSteR documentation (SiSteR_Overview.pdf)
