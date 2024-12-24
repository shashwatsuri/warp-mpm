import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
import torch
import meshio
import trimesh
from mesh_to_sdf import mesh_to_voxels
wp.init()
wp.config.verify_cuda = True
from render_init import Hemisphere_Init

dvc = "cuda:0"


import numpy as np
import trimesh
import scipy

tetra_mesh = meshio.read("/scratch-ssd/Repos/warp-mpm/shapes/hemisphere.vtk")
# # Example usage
# obj_file = "/scratch-ssd/Repos/warp-mpm/shapes/homer.obj"  # Replace with the path to your .obj file
# mesh = trimesh.load(obj_file)
# centroid = mesh.bounding_box.centroid

mpm_solver = MPM_Simulator_WARP(10) # initialize with whatever number is fine. it will be reintialized
multiplier = 8.0 # 7 for homer 8 for hemiphere
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

sim_frames = 300



material_params = {
    'E': 1e4,
    'nu': .3,
    "material": "jelly",
    'friction_angle': 35,
    'g': [0.0, 0.0, 0.0], # -5 for hemisphere -10 for homer
    "density": density
}
mpm_solver.set_parameters_dict(material_params)

mpm_solver.finalize_mu_lam_bulk() # set mu and lambda from the E and nu input


directory_to_save = './sim_results/hemisphere'
if not os.path.exists(directory_to_save):
    os.makedirs(directory_to_save)

stage_path = os.path.join(directory_to_save,"hemisphere_warp.usd")

traj=[]
num_samples = min([len(mpm_solver.mpm_state.particle_x),8_000])
indices = np.random.choice(np.arange(len(mpm_solver.mpm_state.particle_x)),num_samples,replace=False)

trajectories = np.load('/scratch-ssd/Repos/Deformation-Learning/data/hemisphere/trajectories/hemisphere_traj.npy')  # Replace with actual data

positions = mpm_solver.mpm_state.particle_x.numpy()[indices]


# setting up remapping
rotation_matrix = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
],dtype=np.float32)

trajectories  = ((trajectories @ rotation_matrix)/5.0) + offset

point_cloud = trajectories[0]

trimesh.points.PointCloud(point_cloud).export("./shapes/traj.ply")
kdtree = scipy.spatial.cKDTree(positions)
_, known = kdtree.query(point_cloud)  # Find closest mesh vertices
Y = point_cloud  # Target positions
ground_indices = np.where(positions[:,1]<0.1)[0]
known = np.concatenate((known,ground_indices))
Y = np.concatenate((Y,positions[ground_indices]),axis=0)
known, unique_indices = np.unique(known, return_index=True)
Y = Y[unique_indices]

hemisphere_pc = Hemisphere_Init(stage_path,sim_frames,known,trajectories)
mpm_solver.set_trajectory(known,trajectories)

for k in range(sim_frames):
    hemisphere_pc.render(mpm_solver.mpm_state.particle_x.numpy()[known],k)
    mpm_solver.p2g2p(k, 0.002, device=dvc)

if hemisphere_pc.renderer:
    hemisphere_pc.renderer.save()
