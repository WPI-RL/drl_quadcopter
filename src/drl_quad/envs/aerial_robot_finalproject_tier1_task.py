# Modified by: Arjan Gupta & Taylor Bergeron

# Original copyright:
# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of the aeriel_gym_simulator project.

import numpy as np
import os
import torch
import torchvision
from torchvision.transforms import transforms
import xml.etree.ElementTree as ET

from aerial_gym import AERIAL_GYM_ROOT_DIR, AERIAL_GYM_ROOT_DIR

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *
from aerial_gym.envs.base.base_task import BaseTask
from .aerial_robot_finalproject_tier1_cfg import AerialRobotCfgFinalProjectTier1 as AerialRobotCfg
from aerial_gym.envs.controllers.controller import Controller
from aerial_gym.utils.helpers import asset_class_to_AssetOptions
from PIL import Image as im

class AerialRobotFinalProjectTier1(BaseTask):

    def __init__(self, cfg: AerialRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.debug = False
        self.cfg = cfg

        # Override num_envs to 1
        self.cfg.env.num_envs = 1

        self.enable_onboard_cameras = self.cfg.env.enable_onboard_cameras

        self.max_episode_length = int(self.cfg.env.episode_length_s / self.cfg.sim.dt)
        self.debug_viz = False
        num_actors = 1

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless

        # self.env_asset_manager = AssetManager(self.cfg, sim_device)
        self.cam_resolution = (480,270)

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        bodies_per_env = self.robot_num_bodies

        self.vec_root_tensor = gymtorch.wrap_tensor(
            self.root_tensor).view(self.num_envs, num_actors, 13)

        self.root_states = self.vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[..., 0:3]
        self.root_quats = self.root_states[..., 3:7]
        self.root_linvels = self.root_states[..., 7:10]
        self.root_angvels = self.root_states[..., 10:13]

        self.privileged_obs_buf = None
        if self.vec_root_tensor.shape[1] > 1:
            self.env_asset_root_states = self.vec_root_tensor[:, 1:, :]
            if self.get_privileged_obs:
                self.privileged_obs_buf = self.env_asset_root_states
                

        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()
        self.counter = 0
        self.last_reset_counter = 0

        self.action_upper_limits = torch.tensor(
            [1, 1, 1, 1], device=self.device, dtype=torch.float32)
        self.action_lower_limits = torch.tensor(
            [-1, -1, -1, -1], device=self.device, dtype=torch.float32)

        # control tensors
        self.action_input = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3),
                                  dtype=torch.float32, device=self.device, requires_grad=False)
        self.torques = torch.zeros((self.num_envs, bodies_per_env, 3),
                                   dtype=torch.float32, device=self.device, requires_grad=False)

        self.controller = Controller(self.cfg.control, self.device)

        self.full_camera_array = torch.zeros((self.num_envs, 270, 480), device=self.device)

        if self.viewer:
            cam_pos_x, cam_pos_y, cam_pos_z = self.cfg.viewer.pos[0], self.cfg.viewer.pos[1], self.cfg.viewer.pos[2]
            cam_target_x, cam_target_y, cam_target_z = self.cfg.viewer.lookat[0], self.cfg.viewer.lookat[1], self.cfg.viewer.lookat[2]
            cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
            cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
            cam_ref_env = self.cfg.viewer.ref_env
            
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        # To save images
        self.save_images = False

        # Action display fixed coordinate
        self.action_display_fixed_coordinate = torch.tensor([[5,5,5]], device=self.device, dtype=torch.float32)

        # Set drone hit ground buffer
        self.drone_hit_ground_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        self.depth_image = torch.zeros((1, 1024), device=self.device)
        self.contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(self.contact_force_tensor).view(self.num_envs, bodies_per_env, 3)[:,
                              0]
        self.collisions = torch.zeros(self.num_envs, device=self.device)

    def create_sim(self):
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()
        self.progress_buf = torch.zeros(
            self.cfg.env.num_envs, device=self.sim_device, dtype=torch.long)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self):
        print("\nCREATING AerialRobot for RBE 595 Final Project - Tier 1\n")
        asset_path = self.cfg.robot_asset.file.format(
            AERIAL_GYM_ROOT_DIR=AERIAL_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = asset_class_to_AssetOptions(self.cfg.robot_asset)

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)

        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        start_pose = gymapi.Transform()
        self.env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-self.env_spacing, -
                                self.env_spacing, -self.env_spacing)
        env_upper = gymapi.Vec3(
            self.env_spacing, self.env_spacing, self.env_spacing)
        self.actor_handles = []
        self.envs = []
        self.camera_handles = []
        self.camera_tensors = []

        # Set Camera Properties
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = self.cam_resolution[0]
        camera_props.height = self.cam_resolution[1]
        camera_props.far_plane = 15.0
        camera_props.horizontal_fov = 87.0
        # local camera transform
        local_transform = gymapi.Transform()
        # position of the camera relative to the body
        local_transform.p = gymapi.Vec3(0.15, 0.00, 0.05)
        # orientation of the camera relative to the body
        local_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = torch.tensor([0, 0, 0], device=self.device)
            start_pose.p = gymapi.Vec3(*pos)

            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.robot_asset.name, i, self.cfg.robot_asset.collision_mask, 0)
            
            pos = torch.tensor([2, 0, 0], device=self.device)
            wall_pose = gymapi.Transform()
            wall_pose.p = gymapi.Vec3(*pos)
            self.robot_body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.gym.attach_camera_to_body(cam_handle, env_handle, actor_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            self.camera_handles.append(cam_handle)
            camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH)
            torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
            self.camera_tensors.append(torch_cam_tensor)
        
        self.robot_mass = 0
        for prop in self.robot_body_props:
            self.robot_mass += prop.mass
        
    def render_cameras(self):        
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.dump_images()
        self.gym.end_access_image_tensors(self.sim)
        return
    
    def check_collisions(self):
        ones = torch.ones((self.num_envs), device=self.device)
        zeros = torch.zeros((self.num_envs), device=self.device)
        self.collisions[:] = 0
        self.collisions = torch.where(torch.norm(self.contact_forces, dim=1) > 0.1, ones, zeros)

    def dump_images(self):
        for env_id in range(self.num_envs):
            # the depth values are in -ve z axis, so we need to flip it to positive
            self.full_camera_array[env_id] = -self.camera_tensors[env_id]
        return

    def step(self, _position_increment):
        position_increment = _position_increment.to(self.device)

        # Clamp the position_increment by looking at the action limits
        position_increment = tensor_clamp(
            position_increment, self.action_lower_limits[:3], self.action_upper_limits[:3])
        
        # Increment the position with current position
        new_position = position_increment + self.root_positions[0]
        new_position.to(self.device)
        print("new_position",new_position)
        # Increment the position with fixed coordinate
        # position_increment = position_increment + self.action_display_fixed_coordinate[0]
        
        # step physics and render each frame
        for i in range(self.cfg.env.num_control_steps_per_env_step):
            self.pre_physics_step(new_position)
            self.gym.simulate(self.sim)
            # NOTE: as per the isaacgym docs, self.gym.fetch_results must be called after self.gym.simulate, but not having it here seems to work fine
            # it is called in the render function.
            self.post_physics_step()

        self.render(sync_frame_time=False)
        self.render_cameras()
        
        self.progress_buf += 1
        self.compute_observations()
        self.compute_reward()
        self.check_collisions()

        save_images_every = 500

        # Save depth image to file
        if self.save_images and self.counter % save_images_every == 0:
                # print("self.counter:", self.counter)
                self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_DEPTH, "depth_image_"+str(self.counter)+".png")
                self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_COLOR, "rgb_image_"+str(self.counter)+".png")
        
        # FOR TRAINING THE NN (only where images are needed)
        # Store depth image in a buffer
        if self.enable_onboard_cameras:
            # Get the depth image from the camera array
            depth_im = self.full_camera_array[0]

            # The given depth image has shape (270, 480), but we need (1, 1024)
            # So, first we need to scale it to 32x32 on the GPU
            depth_im_pytorch = depth_im.unsqueeze(0).unsqueeze(0)
            depth_im_pytorch = torch.nn.functional.interpolate(depth_im_pytorch, size=(32, 32), mode='bilinear', align_corners=False)

            # Now, the issue is that the depth image has many nan values
            # So, we need to replace them with 0.0
            depth_im_pytorch = torch.where(torch.isnan(depth_im_pytorch), torch.zeros_like(depth_im_pytorch), depth_im_pytorch)

            # Also, the 0-1 range is flipped, so we need to flip it back
            depth_im_pytorch = 1.0 - depth_im_pytorch

            # print("depth_im:", depth_im)
            depth_im=depth_im.detach().cpu().numpy()
            depth_im[depth_im == -np.inf] = 0
            depth_im[depth_im < -10] = -10
            normalized_depth = -255.0*(depth_im/np.min(depth_im + 1e-4))
            normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
            
            normalized_depth_image_resized = normalized_depth_image.resize((32,32))
            depth_im=normalized_depth_image_resized

            # Save the 32x32 depth image to a file after certain number of iterations
            if self.save_images and self.counter % save_images_every == 0:
                torchvision.utils.save_image(depth_im_pytorch, "depth_image_tensor_"+str(self.counter)+".png")
                depth_im.save("depth_env_32x32_%d.jpg"%(self.counter))
                normalized_depth_image.save("depth_env__%d.jpg"%(self.counter))

            # Convert to tensor from numpy
            # Now, we can flatten it to (1, 1024)
            transform_norm = transforms.Compose([
                transforms.Normalize([0.25], [0.25])
            ])
            pil_to_tensor = transforms.ToTensor()(depth_im).unsqueeze_(0)
            d1=transform_norm(pil_to_tensor)

            self.depth_image = pil_to_tensor.flatten()
            # print("self.depth_image:", self.depth_image)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.time_out_buf = self.progress_buf > self.max_episode_length
        self.extras["time_outs"] = self.time_out_buf
        # return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, self.drone_hit_ground_buf
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, self.drone_hit_ground_buf, self.collisions

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.root_states[env_ids,
                         0:3] = 2.0*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        self.root_states[env_ids,
                         7:10] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        self.root_states[env_ids,
                         10:13] = 0.2*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)

        self.root_states[env_ids, 3:7] = 0
        self.root_states[env_ids, 6] = 1.0

        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        self.reset_buf[env_ids] = 1
        self.progress_buf[env_ids] = 0

        print("\n\nResetting env\n\n")

        self.last_reset_counter = self.counter

    def get_current_position(self):
        return self.root_positions

    def get_depth_image(self):
        return self.depth_image

    def pre_physics_step(self, position):
        
        # resets
        if self.debug:
            if self.counter % 250 == 0:
                print("self.counter:", self.counter)
        self.counter += 1

        # Move the position_increment to the device
        

        self.action_input[:] = torch.cat([position, torch.tensor([0], device=self.device)])

        # clear position_increment for reset envs
        self.forces[:] = 0.0
        self.torques[:, :] = 0.0

        output_thrusts_mass_normalized, output_torques_inertia_normalized = self.controller(
            self.root_states, self.action_input)
        self.forces[:, 0, 2] = self.robot_mass * (
            -self.sim_params.gravity.z) * output_thrusts_mass_normalized
        self.torques[:, 0] = output_torques_inertia_normalized
        self.forces = torch.where(self.forces < 0, torch.zeros_like(self.forces), self.forces)

        # apply position_increment
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(
            self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def compute_observations(self):
        self.obs_buf[..., :3] = self.root_positions
        self.obs_buf[..., 3:7] = self.root_quats
        self.obs_buf[..., 7:10] = self.root_linvels
        self.obs_buf[..., 10:13] = self.root_angvels
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.drone_hit_ground_buf[:] = compute_quadcopter_reward(
            self.root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, self.progress_buf, self.max_episode_length,
            (self.counter - self.last_reset_counter)
        )


###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@torch.jit.script
def quat_axis(q, axis=0):
    # type: (Tensor, int) -> Tensor
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


@torch.jit.script
def compute_quadcopter_reward(root_positions, root_quats, root_linvels, root_angvels, reset_buf, progress_buf, max_episode_length, counter):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, int) -> Tuple[Tensor, Tensor, Tensor]

    # distance to target
    target_dist = torch.sqrt(root_positions[..., 0] * root_positions[..., 0] +
                             root_positions[..., 1] * root_positions[..., 1] +
                             (root_positions[..., 2]) * (root_positions[..., 2]))
    pos_reward = 2.0 / (1.0 + target_dist * target_dist)

    dist_reward = (20.0 - target_dist) / 40.0

    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward) + dist_reward

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    # die = torch.where(target_dist > 10.0, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)
    # print("reset1",reset)
    reset = torch.where(torch.norm(root_positions, dim=1) > 35.0, ones, reset) # out of bounds for a norm distance of 20.0
    # print("reset2",reset)
    # Above a certain self.counter number, if the z coordinate is too close to ground, then reset
    if counter > -1:
        ground_threshold = 0.15
        reset = torch.where(root_positions[:, 2] <= ground_threshold, ones, reset)
        reset = torch.where(root_positions[:, 0] >= 20.0, ones, reset)
        reset = torch.where(root_positions[:, 1] >= 20.0, ones, reset)
        # print("reset3",reset)
        drone_hit_ground = torch.where(root_positions[:, 2] <= ground_threshold, ones, die)
    else:
        drone_hit_ground = torch.zeros_like(reset_buf)

    return reward, reset, drone_hit_ground
