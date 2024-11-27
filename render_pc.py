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

# @wp.kernel
# def render_spheres(
#     trajectory: wp.array(dtype=wp.vec3), #type:ignore
#     renderer: wp.sim.render.SimRendererUsd,
# ):
#     tid = wp.tid()
#     renderer.render_sphere(
#                 name="sphere"+str(tid), pos=trajectory[tid],rot = wp.quat_identity(), radius=0.07, color=(1.0, 0.1, 0.1)
#             )

class HemispherePC:
    def __init__(self, stage,sim_frames):
        self.sim_time = 0.0
        self.sim_frames = sim_frames
        self.idx=0
        self.sim_dt = 1.0

        builder = wp.sim.ModelBuilder()
        self.model = builder.finalize()
        self.renderer = wp.sim.render.SimRendererUsd(self.model, stage, scaling=1.0, fps= 1)

    def render(self,trajectory):
        if self.renderer is None:
            return
        self.renderer.begin_frame(self.sim_time)
        # wp.launch(
        #     kernel=render_spheres,
        #     dim=len(trajectory),
        #     inputs=[trajectory,self.renderer]
        # )
        i=0
        for p in trajectory:
            self.renderer.render_sphere(
                name="sphere"+str(i), pos=p,rot = wp.quat_identity(), radius=0.07, color=(1.0, 0.1, 0.1)
            )
            i+=1
        self.renderer.end_frame()
        self.sim_time += self.sim_dt

