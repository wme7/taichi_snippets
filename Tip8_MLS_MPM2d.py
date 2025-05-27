from math import pi
import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)

# Simulation settings
n_grid = 128                  # Number of particles and number of grid cells per direction
dx = 1 / n_grid               # Grid spacing
dt = 2e-4                     # Time step
g = ti.Vector([0, -9.8])      # Gravity

# Sand material
n_particle = 10000                                       # Number of particles
rho_0 = 1500                                             # Particle density
V0_sand = (dx * 0.5) ** 2                                # Particle volume
mass0_sand = rho_0 * V0_sand                             # Particle mass

E, v = 18e6, 0.2                                         # Young's modulus, Poisson's ratio
G, K = int(E/(2*(1+v))), int(E/(3*(1-2*v)))              # Shear modulus, bulk modulus
lambda_0 = int(E * v / ((1 + v) * (1 - 2 * v)))          # Lame parameter, shear modulus
f_a, d_a, cohe = 30 * pi/180, 0.1 * pi/180, 1000         # Friction angle, dilation angle, cohesion
q_f, q_d = 3 * ti.tan(f_a) / ti.sqrt(9 + 12 * ti.tan(f_a)**2), \
           3 * ti.tan(d_a) / ti.sqrt(9 + 12 * ti.tan(d_a)**2)
k_f = 3 * cohe / ti.sqrt(9 + 12 * ti.tan(f_a)**2)
a_B = ti.sqrt(1 + q_f ** 2) - q_f
max_t_s = cohe / ti.tan(f_a)                             # Maximum tensile strength

# Data containers — material points
e = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)                # Strain
e_s = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)              # Deviatoric strain
e_v = ti.field(dtype=float, shape=n_particle)                           # Volumetric strain

delta_e = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)          # Strain increment
delta_e_s = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)        # Deviatoric strain increment
delta_e_v = ti.field(dtype=float, shape=n_particle)                     # Volumetric strain increment

omiga = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)            # Spin tensor

sigma = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)            # Stress
S_s = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)              # Deviatoric stress
S_m = ti.field(dtype=float, shape=n_particle)                           # Mean (spherical) stress — scalar

C_e = ti.Matrix.field(3, 3, float, shape=1)                             # Elastic stiffness matrix, Voigt notation
rho_sand = ti.field(dtype=float, shape=n_particle)                      # Particle density
x = ti.Vector.field(2, dtype=float, shape=n_particle)                   # Particle position
v = ti.Vector.field(2, dtype=float, shape=n_particle)                   # Particle velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)                # Affine velocity field, = velocity increment
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)                # Deformation gradient

# Data containers — grid
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))        # Grid node momentum or velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))                  # Grid node mass
grid_f = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))        # Grid node force


@ti.func
def project(p):                                                             # Plastic projection — project trial stress back to yield surface
        St_s = S_s[p] + (S_s[p] @ omiga[p].transpose() + omiga[p] @ S_s[p]) * dt + 2 * G * delta_e_s[p]  # Trial deviatoric stress
        St_m = S_m[p] + K * delta_e_v[p]                                                                 # Trial mean stress
        St_t = ti.sqrt(0.5 * (St_s[0,0]**2 + St_s[1,1]**2 + 2 * St_s[1,0]**2))                           # Equivalent shear stress
        delta_lam = 0.0                                         # Amount of plastic deformation
        fs = St_t + q_f * St_m - k_f                            # Shear yield function
        hs = St_t - a_B * (St_m - max_t_s)                      # Tensile yield function
        # print(fs, hs, max_t_s)
        if St_m < max_t_s:
            if fs > 0:                                          # Shear failure
                # print("Shear failure")
                delta_lam = fs / (G + K * q_f * q_d)
                S_m[p] = St_m - K * delta_lam * q_d             # Update mean stress
                S_t = k_f - q_f * S_m[p]                        # Update shear stress
                S_s[p] = S_t / St_t * St_s                      # Update deviatoric stress
                sigma[p] = S_s[p] + S_m[p] * ti.Matrix.identity(float, 2)  # Cauchy stress (vector)
            else:                                               # No failure
                # print("No failure")
                S_m[p] = St_m                                   # Update mean stress
                S_s[p] = St_s                                   # Update deviatoric stress
                sigma[p] = S_s[p] + S_m[p] * ti.Matrix.identity(float, 2)  # Cauchy stress (vector)
        elif St_m >= max_t_s:
            if hs > 0:                                          # Shear failure
                # print("Shear failure")
                delta_lam = fs / (G + K * q_f)
                S_m[p] = St_m - K * delta_lam * q_d             # Update mean stress
                S_t = k_f - q_f * S_m[p]                        # Update shear stress
                S_s[p] = S_t / St_t * St_s                      # Update deviatoric stress
                sigma[p] = S_s[p] + S_m[p] * ti.Matrix.identity(float, 2)  # Cauchy stress (vector)
            else:                                               # Tensile failure
                # print("Tensile failure")
                S_m[p] = max_t_s                                # Update mean stress
                S_s[p] = St_s                                   # Update deviatoric stress
                sigma[p] = S_s[p] + S_m[p] * ti.Matrix.identity(float, 2)  # Cauchy stress (vector)


@ti.kernel
def p2g():                                                  # Update grid node force, mass, and momentum based on particle information
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]   # Reset grid momentum
        grid_f[i, j] = [0, 0]   # Reset grid force
        grid_m[i, j] = 0        # Reset grid mass



    for p in x:                                                         # Iterate over all particles
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]        # Deformation gradient matrix
        e_dot = 0.5 * (C[p] + C[p].transpose())                         # Strain rate
        delta_e[p] = e_dot * dt                                         # Strain increment
        omiga[p] = 0.5 * (C[p] - C[p].transpose())                      # Spin rate
        delta_e_v[p] = delta_e[p].trace()                               # Volumetric strain increment
        rho_sand[p] = rho_sand[p] / (1 + delta_e_v[p])                  # Density change (update density according to volumetric strain)

        # print(delta_e_v[p])

        delta_e_s[p] = delta_e[p] - 0.5 * delta_e_v[p] * ti.Matrix.identity(float, 2)             # Deviatoric strain increment
        project(p)
        base = (x[p] / dx - 0.5).cast(int)                              # 3x3 grid base coordinates
        if base[0] < 0 or base[1] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2:          # Protection mechanism to prevent operations on invalid grid nodes
            continue
        fx = x[p] / dx - base.cast(float)                                                         # Relative distance between particle and base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (1.5 - (2 - fx)) ** 2]            # Weights — corresponding to three grids
        grad_w = [fx - 1.5, 2 - 2 * fx, fx - 0.5]                                                 # Weight gradients, todo: apply kernel function gradient when updating grid node force
        for i, j in ti.static(ti.ndrange(3, 3)):                                                  # Iterate over 3x3 grid around particle
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grad_weight = ti.Matrix([grad_w[i][0] * w[j][1] / dx, w[i][0] * grad_w[j][1] / dx])
            # print("Weight:",weight)
            # print("Weight gradient:",grad_weight)
            grid_v[base + offset] += weight * mass0_sand * (v[p] + C[p] @ dpos)              # Grid momentum
            grid_m[base + offset] += weight * mass0_sand                                     # Grid mass
            grid_f[base + offset] += - mass0_sand /rho_sand[p] * sigma[p] @ grad_weight        # Internal force (gravity is updated at grid)
    # print(grid_f[60,10])

@ti.kernel
def update_grid():                                                          # Update velocity based on node force and apply boundary conditions
    for i, j in grid_m:                                                     # Iterate over all grid nodes
        if grid_m[i, j] > 0:
            grid_v[i,j] = (grid_v[i,j] + grid_f[i,j] * dt) / grid_m[i,j]    # Update node velocity
            grid_v[i,j] += dt * (g)           # Velocity change due to gravity and node force

            # print("Grid velocity:", grid_v[i, j])
            # print("Grid force:", grid_f[i, j])

            normal = ti.Vector.zero(float,2)                                # Determine which boundary, for friction
            if i < 3 and grid_v[i, j][0] < 0:                                        # Cancel grid node velocity towards lower boundary to prevent particles from penetrating
                normal = ti.Vector([1, 0])
            if i > n_grid - 3 and grid_v[i, j][0] > 0:                      # Cancel grid node velocity towards upper boundary to prevent particles from penetrating
                normal = ti.Vector([-1, 0])
            if j < 3 and grid_v[i, j][1] < 0:
                normal = ti.Vector([0, 1])
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                normal = ti.Vector([0, -1])

            if not (normal[0] == 0 and normal[1] == 0):                       # Check if on boundary
                s = grid_v[i, j].dot(normal)
                if s <= 0:
                    v_normal = s * normal
                    v_tangent = grid_v[i, j] - v_normal
                    vt = v_tangent.norm()
                    if vt > 1e-12: grid_v[i, j] = v_tangent - (vt if vt < -0.5 * s else -0.5 * s) * (v_tangent / vt)


@ti.kernel
def g2p():                                                                              # Update particle velocity, affine velocity, and position via grid nodes
    for p in x:                                                                         # Iterate over all particles
        base = (x[p] / dx - 0.5).cast(int)                                              #
        fx = x[p] / dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

        new_v = ti.Vector.zero(float, 2)                                                # Container for new velocity of a single particle
        new_C = ti.Matrix.zero(float, 2, 2)                                             # Container for new affine velocity of a single particle
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = (ti.Vector([i, j]).cast(float) - fx) * dx                            # Absolute distance
            g_v = grid_v[base + ti.Vector([i, j])]                                      # Velocity of each grid point in 3x3
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v                                                       # Recalculate particle velocity according to weights of 3x3 grid
            new_C += 4 * weight * g_v.outer_product(dpos) / dx                          # Update C
        v[p], C[p] = new_v, new_C                                                       # Return particle velocity and C
        x[p] += dt * v[p]                                                               # Calculate displacement according to velocity


@ti.kernel
def initialize():
    init_pos = ti.Vector([0.4, 0.3])
    cube_size = 0.2
    spacing = 0.002
    num_per_row = (int) (cube_size // spacing)             # Number of particles per row parallel to x-axis

    for i in range(n_particle):
        floor = i // num_per_row                       # Particle y-coordinate (gravity direction)
        col = i % num_per_row                          # Particle x-coordinate
        x[i] = ti.Vector([col*spacing, floor*spacing]) + init_pos
        v[i] = ti.Matrix([0, 0])
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        rho_sand[i] = rho_0
        sigma[i] = ti.Matrix([[0, 0], [0, 0]])                      # Cauchy stress
        S_m[i] = (sigma[i][0,0] + sigma[i][1,1]) * 0.5              # Spherical stress
        S_s[i] = sigma[i] - S_m[i] * ti.Matrix.identity(float, 2)   # Deviatoric stress


def main():
    initialize()
    gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color = 0xFFFFFF)
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for s in range(int(2e-3 // dt)):                                    # Total time frames
            print("Frame " + str(s))
            p2g()
            update_grid()
            g2p()
        gui.circles(x.to_numpy(), radius=2, color=0x000080,)
        print(v[10])
        for i in np.arange(0, 1, 1/n_grid):
            gui.line([0, i], [1, i], radius=0.8, color=0xD0D0D0)
            gui.line([i, 0], [i, 1], radius=0.8, color=0xD0D0D0)
        gui.show()

if __name__ == "__main__":
    main()