#removes noise variables and resets restitution constants to 0.95

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import pkg_resources
import cv2
import time
import random
import math

class TT3Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        
        # paths to urdfs for pybullet rendering
        self.robot_path = pkg_resources.resource_filename(__name__, 'robot/robot.urdf')
        self.table_path = pkg_resources.resource_filename(__name__, 'table/robot.urdf')
        self.ball_path = pkg_resources.resource_filename(__name__, 'ball/robot.urdf')

        self.render_mode = render_mode
        self.max_steps = 800 # max steps before truncation (defined here instead of during Gym registration)
        self.steps_taken = 0
        self.max_ball_noise = 0.0
        self.max_pose_noise = 0.0
        self.agent_min_ball_depth = 0.3
        self.l_table = 2.74
        self.w_table = 1.525
        self.h_table = 0.1

        # ball trajectory ranges
        self.goal_x = (-1.3, -0.6)
        self.goal_y = (-0.7, 0.7)
        self.start_x = (0.3, 1.6)
        self.start_y = (-0.9, 0.9)
        self.start_z = (0.2, 0.5)
        self.vz_bounds = (2.5,4)

        self.joints = [0, 1, 2, 3, 4, 5]
        self.numJoints = len(self.joints)
        self.frame_skips = 30 # frames rendered by pybullet before agent is queried for a new action
    
        # vars for monitoring during training
        self.episode_count = 0
        self.ball_in_count = 0
        self.ball_touch_count = 0
        self.d2t_sum = 0
        self.ball_in_vec = []

        # pybullet setup
        if self.render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        p.setRealTimeSimulation(0)
        p.resetSimulation()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.numJoints,), dtype=float)

    def get_obs(self):

        joint_positions = [p.getJointState(self.robot, joint)[0] for joint in self.joints]
        joint_velocities = [p.getJointState(self.robot, joint)[1] for joint in self.joints]
        ball_position = list(p.getBasePositionAndOrientation(self.ball)[0])
        ball_velocity = list(p.getBaseVelocity(self.ball)[0])
        current_obs = np.array(joint_positions+joint_velocities+ball_position+ball_velocity)
        return current_obs
    
    def reset(self,seed=None, options = None): 
        super().reset(seed=seed) # setting random seed for environment
        #reset simulation vars and entities
        self.episode_count += 1
        self.terminated = False
        self.truncated = False
        self.steps_taken = 0
        self.state = 0
        self.min_r2b = 10
        self.prevent_bug = 0
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        self.plane = p.loadURDF("plane.urdf") 
        self.robot = p.loadURDF(self.robot_path, [-1.8,0,0], useFixedBase = 1)
        self.table = p.loadURDF(self.table_path, [0,0,0],useFixedBase = 1)

        # apply random noise to robot pose
        poses = [random.uniform(-self.max_pose_noise, self.max_pose_noise) for _ in range(self.numJoints)]
        p.setJointMotorControlArray(
            bodyIndex = self.robot, 
            jointIndices = self.joints, 
            controlMode = p.POSITION_CONTROL, 
            targetPositions = poses,
            forces = [500]*self.numJoints)
        for _ in range(50):
            p.stepSimulation()

        #load in the ball, and assign initial velocity
        start, v = self.get_trajectory()
        self.ball = p.loadURDF(self.ball_path, start)
        p.resetBaseVelocity(self.ball, linearVelocity = v)

        #make the paddle and table bouncy
        p.changeDynamics(self.ball, -1, restitution = 0.95)
        p.changeDynamics(self.table, -1, restitution = 0.95)
        p.changeDynamics(self.robot, 2, restitution = 0.95)

        observation = self.get_obs()
        info = {"info":"info"} # dict info return required by Gym

        return observation, info
    
    def get_trajectory(self):

        goal = [random.uniform(*self.goal_x), random.uniform(*self.goal_y), self.h_table]
        start = [random.uniform(*self.start_x), random.uniform(*self.start_y), random.uniform(*self.start_z)]
        vz = random.uniform(*self.vz_bounds)

        t = (vz + math.sqrt(vz**2 + 19.6*(start[2]+goal[2])))/9.8
        vx = (goal[0] - start[0])/t
        vy = (goal[1] - start[1])/t
        v = [vx,vy,vz]
        return start, v

    # helper function for euclidean distance
    def d_euc(self, point1, point2):
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))
    
    # distance between point (x,y) and midpoint of opponent's side of table
    def d2m(self,x,y): 
        return self.d_euc([x,y],[0.8,0])
    
    # distance between point (x,y) and opponent's side of table
    def d2t(self,x,y): 
        if x < self.agent_min_ball_depth:
            dx = abs(self.agent_min_ball_depth - x) # min ball depth prevents agent from learning to hit the net
        elif x > self.l_table/2:
            dx = x - self.l_table/2
        else:
            dx = 0
        if y > self.w_table/2:
            dy = y - self.w_table
        elif y < -self.w_table/2:
            dy = self.w_table - y
        else:
            dy = 0
        return math.sqrt(dx**2 + dy**2)
    
    # distance between robot end-effector and ball
    def r2b(self):
        rob = list(p.getLinkState(self.robot, self.numJoints-1)[0])
        ball = list(p.getBasePositionAndOrientation(self.ball)[0])
        return self.d_euc(rob, ball)
    
    def reset_pose(self):
        p.setJointMotorControlArray(
            bodyIndex = self.robot, 
            jointIndices = self.joints, 
            controlMode = p.POSITION_CONTROL, 
            targetPositions = [0]*self.numJoints,
            targetVelocities = [5]*self.numJoints,
            forces = [500]*self.numJoints)

    def get_reward(self):
        if self.state == 0: # up until the ball bounces on table
            self.prevent_bug +=1
            if p.getContactPoints(self.table, self.ball) and self.prevent_bug > 5:
                self.state = 1
            elif p.getContactPoints(self.robot, self.ball):
                self.terminated = True
                #print("hitting ball too early!")
                return -1
        elif self.state == 1: # up until robot hits ball
            r2b = self.r2b()
            if r2b < self.min_r2b:
                self.min_r2b = r2b
            if p.getContactPoints(self.robot, self.ball, self.numJoints-1): # paddle makes contact with ball
                self.reset_timer = 0
                self.state = 2
                self.ball_touch_count += 1
                return 1
            elif p.getContactPoints(self.ball):
                self.terminated = True
                return -self.min_r2b
        elif self.state == 2: # up until ball lands
            point = p.getContactPoints(self.ball, self.plane) or p.getContactPoints(self.ball, self.table)
            while not point:
                self.reset_timer +=1
                self.steps_taken += 1
                # ______________________________
                if self.steps_taken >= self.max_steps:
                    self.truncated = True
                    return 0
                # ______________________________
                if self.reset_timer > 50:
                    self.reset_pose()
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(0.0001) # small delay for better visualization
                point = p.getContactPoints(self.ball, self.plane) or p.getContactPoints(self.ball, self.table)
            point = p.getContactPoints(self.ball)
            d2t = self.d2t(point[0][5][0],point[0][5][1])
            d2m = self.d2m(point[0][5][0],point[0][5][1])
            self.d2t_sum += d2t
            self.terminated = True
            if d2t == 0:
                self.ball_in_count += 1
                return 10 - d2m
            else:
                return max(-d2m/3, -1)
        return 0

    def step(self, action):
        # actions bound to (-1,1), so scale back up to more realistic values
        actions = action*5
        # set target velocities with infinite force
        p.setJointMotorControlArray(
            bodyUniqueId = self.robot, 
            jointIndices = self.joints, 
            controlMode = p.VELOCITY_CONTROL, 
            targetVelocities = actions,
            forces = [500]*self.numJoints)
        
        # apply random noise to ball
        x = random.uniform(-self.max_ball_noise, self.max_ball_noise)
        y = random.uniform(-self.max_ball_noise, self.max_ball_noise)
        z = random.uniform(-self.max_ball_noise, self.max_ball_noise)
        p.applyExternalForce(self.ball, -1, [x,y,z], [0,0,0], p.LINK_FRAME)

        # step simulation by number of frame skips
        reward = 0
        for _ in range(self.frame_skips):
            p.stepSimulation()
            self.steps_taken += 1
            if self.steps_taken >= self.max_steps:
                self.truncated = True
                #print(self.state)
                #print("truncating (not good)")

            # for monitoring training
            if self.episode_count >= 10000:
                print("balls in / 10000 = ", self.ball_in_count)
                self.ball_in_vec += [self.ball_in_count]
                print("balls touched / 10000 = ", self.ball_touch_count)
                print("average d2t for touched balls = ", 
                      self.d2t_sum/(self.ball_touch_count+ 0.0001)) # inhibit division by 0
                self.episode_count = 0
                self.ball_in_count = 0
                self.ball_touch_count = 0
                self.d2t_sum = 0
            
            #control_cost = - self.d_euc(targetPos, self.prevPos)/20
            reward += self.get_reward() #- control_cost

            if self.render_mode == "human":
                self.render_frame()
            
            # break out of loop if episode ends
            if self.truncated or self.terminated:
                #print("episode ended")
                break

        observation = self.get_obs()
        info = {"info":"hi"}

        return observation, reward, self.terminated, self.truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_frame()
    
    # self.step() function handles rendering for "human" mode 
    # For "rgb_array" mode, return static image of scene
    def render_frame(self):
        if self.render_mode == "human":
            time.sleep(0.0001) # small delay for better visualization

        if self.render_mode == "rgb_array":
            focus_position,_ = p.getBasePositionAndOrientation(self.robot)
            focus_position = tuple([focus_position[0] + 0.2,focus_position[1] + 0.2,focus_position[2]])
            p.resetDebugVisualizerCamera(
                cameraDistance=0.9, 
                cameraYaw = 40, 
                cameraPitch = -12, 
                cameraTargetPosition = focus_position
            )

            h,w = 4000, 4000
            image = np.array(p.getCameraImage(h, w)[2]).reshape(h,w,4)
            image = image[:, :, :3]
            image = cv2.convertScaleAbs(image)
            return image
    
    # parent class requires override
    def close(self):
        print(self.ball_in_vec)
        return

