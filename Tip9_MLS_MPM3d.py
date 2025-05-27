from math import pi
import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)

# Simulation settings
n_grid = 100                   # Number of particles and number of grid cells per direction
dx = 1 / n_grid                # Grid spacing
dt = 1e-5                      # Time step increment—must be small enough in 3D, otherwise it won't converge
g = ti.Vector([0, -9.8, 0])    # Gravity

# Sand material properties
n_particle = 64000                                       # Number of particles
rho_0 = 1500                                             # Particle density
V0_sand = (dx * 0.5) ** 3                                # Particle volume
mass0_sand = rho_0 * V0_sand                             # Particle mass

E, v = 20e6, 0.3                                         # Young's modulus, Poisson's ratio
G, K = int(E/(2*(1+v))), int(E/(3*(1-2*v)))              # Shear modulus, bulk modulus
lambda_0 = int(E * v / ((1 + v) * (1 - 2 * v)))          # Lame parameter, shear modulus
f_a, d_a, cohe = 40 * pi/180, 10 * pi/180, 10            # Friction angle, dilation angle, cohesion
q_f, q_d = 3 * ti.tan(f_a) / ti.sqrt(9 + 12 * ti.tan(f_a)**2), \
           3 * ti.tan(d_a) / ti.sqrt(9 + 12 * ti.tan(d_a)**2)
k_f = 3 * cohe / ti.sqrt(9 + 12 * ti.tan(f_a)**2)
a_B = ti.sqrt(1 + q_f ** 2) - q_f
max_t_s = cohe / ti.tan(f_a)                             # Maximum tensile strength

# Data containers—material points
e = ti.Matrix.field(3, 3, dtype=float, shape=n_particle)                # Strain
e_s = ti.Matrix.field(3, 3, dtype=float, shape=n_particle)              # Deviatoric strain
e_v = ti.field(dtype=float, shape=n_particle)                           # Volumetric strain

delta_e = ti.Matrix.field(3, 3, dtype=float, shape=n_particle)          # Strain increment
delta_e_s = ti.Matrix.field(3, 3, dtype=float, shape=n_particle)        # Deviatoric strain increment
delta_e_v = ti.field(dtype=float, shape=n_particle)                     # Volumetric strain increment

omiga = ti.Matrix.field(3, 3, dtype=float, shape=n_particle)            # Spin tensor

sigma = ti.Matrix.field(3, 3, dtype=float, shape=n_particle)            # Stress
S_s = ti.Matrix.field(3, 3, dtype=float, shape=n_particle)              # Deviatoric stress
S_m = ti.field(dtype=float, shape=n_particle)                           # Spherical stress—scalar

C_e = ti.Matrix.field(3, 3, float, shape=1)                             # Elastic stiffness matrix, Voigt notation
rho_sand = ti.field(dtype=float, shape=n_particle)                      # Particle density
x = ti.Vector.field(3, dtype=float, shape=n_particle)                   # Particle position
v = ti.Vector.field(3, dtype=float, shape=n_particle)                   # Particle velocity
C = ti.Matrix.field(3, 3, dtype=float, shape=n_particle)                # Affine velocity field, = velocity increment
F = ti.Matrix.field(3, 3, dtype=float, shape=n_particle)                # Deformation gradient

# Data containers—grid
grid_v = ti.Vector.field(3, dtype=float, shape=(n_grid, n_grid, n_grid))        # Grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))                  # Grid node mass
grid_f = ti.Vector.field(3, dtype=float, shape=(n_grid, n_grid, n_grid))        # Grid node force

box_vertex_point = ti.Vector.field(3, dtype=ti.f32, shape=8)        # The coordinates of the 8 corner points of the domain box
box_vertex_point[0] = [0., 0., 0.]
box_vertex_point[1] = [0., 0.5, 0.]
box_vertex_point[2] = [0.5, 0., 0.]
box_vertex_point[3] = [0.5, 0.5, 0.]

box_vertex_point[4] = [0., 0., 0.5]
box_vertex_point[5] = [0., 0.5, 0.5]
box_vertex_point[6] = [0.5, 0., 0.5]
box_vertex_point[7] = [0.5, 0.5, 0.5]

box_edge_index = ti.field(dtype=ti.i32, shape=24)
for i, idx in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):  # Iterate over the indices and elements at the same time
    box_edge_index[i] = idx 


@ti.func
def project(p):                                                             # Plastic projection—pull trial stress back to yield surface
        St_s = S_s[p] + (S_s[p] @ omiga[p].transpose() + omiga[p] @ S_s[p]) * dt + 2 * G * delta_e_s[p]  # Trial deviatoric stress
        St_m = S_m[p] + K * delta_e_v[p]                                                                 # Trial spherical stress
        St_t = ti.sqrt(0.5 * (St_s[0,0]**2 + St_s[1,1]**2 + St_s[2,2]**2 + 2 * (St_s[1,0]**2 + St_s[2,0]**2+ St_s[2,1]**2)))  # Equivalent shear stress
        delta_lam = 0.0                                         # Amount of plastic deformation
        fs = St_t + q_f * St_m - k_f                            # Shear yield equation
        hs = St_t - a_B * (St_m - max_t_s)                      # Tensile yield equation
        # print(fs, hs, max_t_s)
        if St_m < max_t_s:
            if fs > 0:                                          # Shear failure
                # print("Shear failure")
                delta_lam = fs / (G + K * q_f * q_d)
                S_m[p] = St_m - K * delta_lam * q_d             # Update spherical stress
                S_t = k_f - q_f * S_m[p]                        # Update shear stress
                S_s[p] = S_t / St_t * St_s                      # Update deviatoric stress
                sigma[p] = S_s[p] + S_m[p] * ti.Matrix.identity(float, 3)  # Cauchy stress (vector)
            else:                                               # No failure occurred
                # print("No failure")
                S_m[p] = St_m                                   # Update spherical stress
                S_s[p] = St_s                                   # Update deviatoric stress
                sigma[p] = S_s[p] + S_m[p] * ti.Matrix.identity(float, 3)  # Cauchy stress (vector)
        elif St_m >= max_t_s:
            if hs > 0:                                          # Shear failure
                # print("Shear failure")
                delta_lam = fs / (G + K * q_f * q_d)
                S_m[p] = St_m - K * delta_lam * q_d             # Update spherical stress
                S_t = k_f - q_f * S_m[p]                        # Update shear stress
                S_s[p] = S_t / St_t * St_s                      # Update deviatoric stress
                sigma[p] = S_s[p] + S_m[p] * ti.Matrix.identity(float, 3)  # Cauchy stress (vector)
            else:                                               # Tensile failure
                # print("Tensile failure")
                S_m[p] = max_t_s                                # Update spherical stress
                S_s[p] = St_s                                   # Update deviatoric stress
                sigma[p] = S_s[p] + S_m[p] * ti.Matrix.identity(float, 3)  # Cauchy stress (vector)


@ti.kernel
def p2g():                                                  # Update grid node forces, masses, and momenta from particle information
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]   # Reset grid momentum
        grid_f[i, j, k] = [0, 0, 0]   # Reset grid force
        grid_m[i, j, k] = 0           # Reset grid mass

    for p in x:                                                         # Iterate over all particles
        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]        # Deformation gradient matrix
        e_dot = 0.5 * (C[p] + C[p].transpose())                         # Strain rate
        delta_e[p] = e_dot * dt                                         # Strain increment
        omiga[p] = 0.5 * (C[p] - C[p].transpose())                      # Spin rate
        delta_e_v[p] = delta_e[p].trace()                               # Volumetric strain increment
        rho_sand[p] = rho_sand[p] / (1 + delta_e_v[p])                  # Density change (update density according to volumetric strain)

        # print(delta_e_v[p])

        delta_e_s[p] = delta_e[p] - 1/3 * delta_e_v[p] * ti.Matrix.identity(float, 3)             # Deviatoric strain increment
        project(p)
        base = (x[p] / dx - 0.5).cast(int)                              # 3x3 grid base point coordinates
        if base[0] < 0 or base[1] < 0 or base[2] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2 or base[2] >= n_grid - 2: # Protection mechanism, prevent computation on invalid nodes
            continue
        fx = x[p] / dx - base.cast(float)                                                         # Relative distance from particle to base point
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (1.5 - (2 - fx)) ** 2]            # Weights—corresponding to three grids
        grad_w = [fx - 1.5, 2 - 2 * fx, fx - 0.5]                                                 # Weight gradients TODO: Kernel function gradient is used when updating grid node force
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):                                            # Iterate over the 3x3 grids around each particle
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grad_weight = ti.Matrix([grad_w[i][0] * w[j][1] * w[k][2] / dx, w[i][0] * grad_w[j][1] * w[k][2] / dx, w[i][0] * w[j][1] * grad_w[k][2] / dx])
            # print("Weight:",weight)
            # print("Weight gradient:",grad_weight)
            grid_v[base + offset] += weight * mass0_sand * (v[p] + C[p] @ dpos)              # Grid momentum
            grid_m[base + offset] += weight * mass0_sand                                     # Grid mass
            grid_f[base + offset] += - mass0_sand /rho_sand[p] * sigma[p] @ grad_weight      # Internal force (gravity is updated at the grid)


@ti.kernel
def update_grid():                                                              # Update velocity based on node forces and apply boundary conditions
    for i, j, k in grid_m:                                                      # Iterate over all grid nodes
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] = (grid_v[i, j, k] + grid_f[i, j, k] * dt) / grid_m[i, j, k]    # Update node velocity
            grid_v[i, j, k] += dt * g                 # Velocity change due to gravity and node force

            # print("Grid velocity:", grid_v[i, j, k])
            # print("Grid force:", grid_f[i, j, k])

            normal = ti.Vector.zero(float,3)                                # Determine which boundary, used to implement friction
            if i < 3 and grid_v[i, j, k][0] < 0:                                        # Cancel the downward velocity of grid particles at the lower boundary to prevent penetration
                normal = ti.Vector([1, 0, 0])
            if i > n_grid - 3 and grid_v[i, j, k][0] > 0:                      # Cancel the upward velocity of grid particles at the upper boundary to prevent penetration
                normal = ti.Vector([-1, 0, 0])
            if j < 3 and grid_v[i, j, k][1] < 0:
                normal = ti.Vector([0, 1, 0])
            if j > n_grid - 3 and grid_v[i, j, k][1] > 0:
                normal = ti.Vector([0, -1, 0])
            if k < 3 and grid_v[i, j, k][2] < 0:
                normal = ti.Vector([0, 0, 1])
            if k > n_grid - 3 and grid_v[i, j, k][2] > 0:
                normal = ti.Vector([0, 0, -1])

            if not (normal[0] == 0 and normal[1] == 0 and normal[2] == 0):                       # Check if on the boundary
                s = grid_v[i, j, k].dot(normal)
                if s <= 0:
                    v_normal = s * normal
                    v_tangent = grid_v[i, j, k] - v_normal
                    vt = v_tangent.norm()
                    if vt > 1e-12: grid_v[i, j, k] = v_tangent - (vt if vt < -0.2 * s else -0.2 * s) * (v_tangent / vt)


@ti.kernel
def g2p():                                                                              # Update particle velocity, affine velocity, and position from grid nodes
    for p in x:                                                                         # Iterate over all particles
        base = (x[p] / dx - 0.5).cast(int)                                              #
        fx = x[p] / dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

        new_v = ti.Vector.zero(float, 3)                                                # New velocity container for each particle
        new_C = ti.Matrix.zero(float, 3, 3)                                             # New affine velocity container for each particle
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            dpos = (ti.Vector([i, j, k]).cast(float) - fx) * dx                         # Absolute distance
            g_v = grid_v[base + ti.Vector([i, j, k])]                                   # Velocity of each grid point in 3x3 neighborhood
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v                                                       # Recalculate particle velocity from 3x3 grid weights
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2                       # Update C
        v[p], C[p] = new_v, new_C                                                       # Return particle velocity and C
        x[p] += dt * v[p]                                                               # Update displacement from velocity



@ti.kernel
def initialize():
    init_pos = ti.Vector([0.1, 0.03, 0.1])
    cube_size = 0.2
    spacing = 0.005
    num_per_row = (int) (cube_size // spacing)          # Number of particles per row along the x-axis
    num_per_floor = num_per_row * num_per_row              # Total number of particles on the bottom plane
    for i in range(n_particle):
        floor = i // (num_per_floor)                       # Particle y coordinate (gravity direction)
        row = (i % num_per_floor) // num_per_row           # Particle z coordinate
        col = (i % num_per_floor) % num_per_row            # Particle x coordinate
        x[i] = ti.Vector([col*spacing, floor*spacing, row*spacing]) + init_pos
        v[i] = ti.Matrix([-0.0, -0.8, 0])
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rho_sand[i] = rho_0
        sigma[i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])             # Cauchy stress
        S_m[i] = (sigma[i][0,0] + sigma[i][1,1] + sigma[i][2,2]) / 3        # Spherical stress
        S_s[i] = sigma[i] - S_m[i] * ti.Matrix.identity(float, 3)           # Deviatoric stress


@ti.kernel
def initialize_random():
    for i in range(n_particle):
        x[i] = [ti.random() * 0.1 + 0.2, ti.random() * 0.1 + 0.04, ti.random() * 0.1 + 0.2]
        v[i] = ti.Matrix([-3, -3, 0])
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rho_sand[i] = rho_0
        sigma[i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Cauchy stress
        S_m[i] = (sigma[i][0, 0] + sigma[i][1, 1] + sigma[i][2, 2]) / 3  # Spherical stress
        S_s[i] = sigma[i] - S_m[i] * ti.Matrix.identity(float, 3)  # Deviatoric stress


def main():
    initialize()
    window = ti.ui.Window("3d MLS-MPM Zhangxiong", (768, 768))
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0.8, 0.1, 0.7)  # Camera position coordinates
    camera.lookat(0, 0.1, 0.1)
    scene.set_camera(camera)
    while window.running:
        for s in range(int(2e-3 // dt)):                                    # Total frames
            p2g()
            update_grid()
            g2p()
        scene.point_light((1, 1, 1), color=(1, 1, 1))
        scene.particles(x, color=(0.4, 0.1, 0.6), radius=0.0025)
        scene.lines(box_vertex_point, width=1.0, indices=box_edge_index, color=(0, 0, 0))  # Draw the boundary box using lines

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()