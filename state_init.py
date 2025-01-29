import warp as wp
import torch
import meshio
import trimesh
import numpy as np
import scipy
from scipy.spatial import ConvexHull

from render_init import Hemisphere_Init
from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
from scipy.interpolate import RBFInterpolator

wp.init()
wp.config.verify_cuda = True
dvc = "cuda:0"
scale=5.0

tetra_mesh = meshio.read("/scratch-ssd/Repos/warp-mpm/shapes/hemisphere.vtk")
state_mesh = np.load("/scratch-ssd/Repos/Deformation-Learning/data/hemisphere/states/state_00420.npz")["particle_q"]/scale
hull = ConvexHull(state_mesh)

mpm_solver = MPM_Simulator_WARP(10) # initialize with whatever number is fine. it will be reintialized
multiplier = 10.0 # 7 for homer 8 for hemiphere
offset = multiplier/2.0


# You can either load sampling data from an external h5 file, containing initial position (n,3) and particle_volume (n,)
# mpm_solver.load_from_sampling("sand_column.h5", n_grid = 150, device=dvc) 
# mesh is in (-1,1) hence setting grid_lim to 2.0 and translating object by 1.0 to get it in (0,grid_lim)
# tensor_x = torch.asarray(np.array(offset+ tetra_mesh.points,dtype=np.float32))
tensor_x = torch.asarray(np.array(offset+ state_mesh,dtype=np.float32))


mpm_solver.load_initial_data_from_torch(tensor_x=tensor_x,
                                        tensor_volume=torch.ones(len(state_mesh)) * 2.5e-1,
                                        n_grid=150,
                                        grid_lim=multiplier,
                                        device=dvc,
                                        velocity=wp.vec3f(0.0,0.0,0.0))

# Note: You must provide 'density=..' to set particle_mass = density * particle_volume


density=200.0
k_mu=900.00 
k_lambda=500.0
k_damp=0.0

nu = k_lambda/(2*(k_lambda+k_mu))
E = 2*k_mu*(1+nu)




material_params = {
    'E': 1e4,
    'nu': .3,
    "material": "plasticine",
    'friction_angle': 35,
    'g': [0.0, -10.0, 0.0], # -5 for hemisphere -10 for homer
    "density": density
}
mpm_solver.set_parameters_dict(material_params)

mpm_solver.finalize_mu_lam_bulk() # set mu and lambda from the E and nu input


directory_to_save = './sim_results/hemisphere'
if not os.path.exists(directory_to_save):
    os.makedirs(directory_to_save)

stage_path = os.path.join(directory_to_save,"hemisphere_warp_2.usd")

traj=[]
num_samples = min([len(mpm_solver.mpm_state.particle_x),8_000])

trajectories = np.load('/scratch-ssd/Repos/Deformation-Learning/data/hemisphere/trajectories/hemisphere_traj.npy')/scale  # Replace with actual data
sim_frames = trajectories.shape[0]


positions = mpm_solver.mpm_state.particle_x.numpy()[hull.vertices]

# setting up remapping
rotation_matrix = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
],dtype=np.float32)

trajectories  = ((trajectories @ rotation_matrix)) + offset

point_cloud = trajectories[0]

trimesh.points.PointCloud(point_cloud).export("./shapes/traj.ply")
kdtree = scipy.spatial.cKDTree(positions)
_, known = kdtree.query(point_cloud)  # Find closest mesh vertices
Y = point_cloud  # Target positions
# ground_indices = np.where(positions[:,1]<0.1)[0]
# known = np.concatenate((known,ground_indices))
# Y = np.concatenate((Y,positions[ground_indices]),axis=0)
known, unique_indices = np.unique(known, return_index=True)
Y = Y[unique_indices]

# Compute displacement vectors
D = Y - positions[unique_indices]

# Train RBF interpolator on the displacements
rbf = RBFInterpolator(positions[unique_indices], D, kernel='thin_plate_spline')

# Interpolate displacement for all points
displacement = rbf(positions)

# Apply deformation

mask = np.zeros(state_mesh[:,0].shape)
mask[hull.vertices] = 1

traj_pos = positions+displacement

# not_known = np.delete(np.arange(mpm_solver.mpm_state.particle_x.numpy().shape[0]),hull.vertices)

hemisphere_pc = Hemisphere_Init(stage_path,sim_frames,hull.vertices,traj_pos)
mpm_solver.set_trajectory(wp.array(mask,dtype=int),wp.array(traj_pos,dtype=wp.vec3f))

mpm_solver.add_surface_collider((0.0, offset, 0.0), (0.0,trajectories[:,:,1].min()/scale,0.0), 'cut', 0.0)

for k in range(370):
    hemisphere_pc.render(mpm_solver.mpm_state.particle_x.numpy()[hull.vertices],k)
    mpm_solver.p2g2p(k, 0.0002, device=dvc)

if hemisphere_pc.renderer:
    hemisphere_pc.renderer.save()
