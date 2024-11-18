# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import pygalmesh
import meshio
import warp as wp
import warp.sim
import warp.sim.render
import os
import sys
import math
import time
from interval import Interval
import numpy as np

np.random.seed(450)
wp.init()

class Grid:
    def __init__(self, stage, grid_dx, n_grid):
        self.grid_dx = grid_dx
        self.n_grid = n_grid
        self.sim_time=0.0
        builder = wp.sim.ModelBuilder()
        self.model = builder.finalize()
        self.renderer = wp.sim.render.SimRendererUsd(self.model, stage, scaling=1.0, fps= 1)

    def render(self):
        if self.renderer is None:
            return
        self.renderer.begin_frame(self.sim_time)
        self.renderer.render_sphere(
                name="sphere"+str(i), pos=self.trajectories[self.idx,i],rot = wp.quat_identity(), radius=0.07, color=(1.0, 0.1, 0.1)
            )
            
        self.renderer.end_frame()

            


if __name__ == "__main__":
    stage_path = "usds/"+ "c3t1_masks" + '.usd'
    traj_path = "/scratch-ssd/Repos/deformgs/output/hemisphere/c3t1_masks/train/ours_14000/all_trajs.npy"
    grid = Grid(stage_path,traj_path)
    grid.render()

    if grid.renderer:
        grid.renderer.save()