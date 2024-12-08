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
from warp_utils import Dirichlet_collider,SDF_Collider

np.random.seed(450)
wp.init()
dvc="cuda:0"

class HemispherePC:
    def __init__(self, stage,sim_frames,collider_params,offset):
        self.sim_time = 0.0
        self.sim_frames = sim_frames
        self.idx=0
        self.sim_dt = 0.02
        self.collider_params = collider_params

        builder = wp.sim.ModelBuilder(gravity=0.0)
        self.model = builder.finalize(device=dvc)
        self.model.ground= False
        self.renderer = wp.sim.render.SimRendererUsd(self.model, stage, scaling=1.0, fps= 30)
        self.offset = offset

    def render(self,trajectory):
        if self.renderer is None:
            return
        self.renderer.begin_frame(self.sim_time)
        for collider in self.collider_params:
            if "Dirichlet_collider" in str(collider):
                 self.renderer.render_plane(
                      name="mpm_plane_collider",
                      pos=collider.point,
                      rot=wp.quat(1.0, 0.0, 0.0, 0.0),
                      width=10.0,
                      length=10.0
                 )
            else:
                 self.renderer.render_sphere(
                      name="mpm_sphere_collider",
                      pos=collider.pos,
                      rot= wp.quat_identity(),
                      radius=collider.radius
                 )
        self.renderer.render_points(
                name="mpm_points", points=trajectory, radius=0.007, colors=(0.8, 0.4, 0.2)
            )
        self.renderer.end_frame()
        self.sim_time += self.sim_dt

