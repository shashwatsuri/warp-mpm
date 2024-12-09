
import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
import torch
import meshio
wp.init()
wp.config.verify_cuda = True
from render_pc import HemispherePC

dvc = "cuda:0"

tetra_mesh = meshio.read("/scratch-ssd/Repos/Deformation-Learning/shapes/hemisphere.vtk")
mpm_solver = MPM_Simulator_WARP(10) # initialize with whatever number is fine. it will be reintialized
multiplier = 8.0
offset = multiplier/2.0


# You can either load sampling data from an external h5 file, containing initial position (n,3) and particle_volume (n,)
# mpm_solver.load_from_sampling("sand_column.h5", n_grid = 150, device=dvc) 
# mesh is in (-1,1) hence setting grid_lim to 2.0 and translating object by 1.0 to get it in (0,grid_lim)
tensor_x = torch.asarray(np.array(offset+ tetra_mesh.points,dtype=np.float32))



mpm_solver.load_initial_data_from_torch(tensor_x=tensor_x,
                                        tensor_volume=torch.ones(len(tetra_mesh.points)) * 2.5e-8,
                                        n_grid=150,
                                        grid_lim=multiplier,
                                        device=dvc,
                                        velocity=wp.vec3f(0.0,0.0,0.0))

# Note: You must provide 'density=..' to set particle_mass = density * particle_volume


density=100.0
k_mu=9000.00 
k_lambda=5000.0
k_damp=300.0

nu = k_lambda/(2*(k_lambda+k_mu))
E = 2*k_mu*(1+nu)

sim_frames = 400



material_params = {
    'E': 1e4,
    'nu': .3,
    "material": "jelly",
    'friction_angle': 35,
    'g': [0.0, -10.0, 0.0],
    "density": density
}
mpm_solver.set_parameters_dict(material_params)

mpm_solver.finalize_mu_lam_bulk() # set mu and lambda from the E and nu input

# mpm_solver.add_surface_collider((0.0, 4.0, 0.0), (0.0,1.0,0.0), 'sticky', 0.0)

mpm_solver.add_sdf_collider(center=(4.0,2.0,4.0),radius=0.2,surface='sticky',friction=0.0)



directory_to_save = './sim_results/hemisphere'
if not os.path.exists(directory_to_save):
    os.makedirs(directory_to_save)

stage_path = os.path.join(directory_to_save,"hemisphere.usd")
hemisphere_pc = HemispherePC(stage_path,sim_frames,mpm_solver.collider_params)

traj=[]
indices = np.random.choice(np.arange(len(mpm_solver.mpm_state.particle_x)),8_000,replace=False)
for k in range(sim_frames):
    # if (sim_frames%2==0):
    hemisphere_pc.render(mpm_solver.mpm_state.particle_x.numpy()[indices])
    mpm_solver.p2g2p(k, 0.002, device=dvc)

if hemisphere_pc.renderer:
    hemisphere_pc.renderer.save()