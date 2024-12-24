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
from warp_utils import Dirichlet_collider,Sphere_Collider

np.random.seed(450)
wp.init()
dvc="cuda:0"

class Hemisphere_Init:
    def __init__(self, stage,sim_frames,known, trajectories):
        self.sim_time = 0.0
        self.sim_frames = sim_frames
        self.idx=0
        self.sim_dt = 0.02
        self.known = known
        self.trajectories = trajectories
       

        builder = wp.sim.ModelBuilder(gravity=0.0)
        self.model = builder.finalize(device=dvc)
        self.model.ground= False
        self.renderer = wp.sim.render.SimRendererUsd(self.model, stage, scaling=1.0, fps= 30)

    def render(self,trajectory,k):
        if self.renderer is None:
            return
        self.renderer.begin_frame(self.sim_time)
            
        self.renderer.render_points(
                name="mpm_points", points=trajectory, radius=0.007, colors=(0.8, 0.4, 0.2)
            )
        
        self.renderer.render_points(
                name="traj_points", points=self.trajectories[k], radius=0.007, colors=(0.8, 0.4, 0.2)
            )
        
        self.renderer.end_frame()
        self.sim_time += self.sim_dt

