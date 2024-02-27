"""
This is the reward system for the quadcopter in the final project, as
given in Camci et al. 2020.
"""
import torch

class QuadRewardSystem:
    def __init__(self):
        # Initialize reward variables from paper
        
        # For time step distance calculation
        self.dt_end = 0.0
        self.dt_start = 0.0
        self.delta_d = self.dt_end - self.dt_start

        # Lower and upper limits on delta_d
        self.delta_d_l = 0.0
        self.delta_d_u = 1.0

        # Maximum deviation distance
        self.d_max = 19.0

        # Lower and upper limits on reward
        self.R_l = 0.0
        self.R_u = 1.0

        # Mild punishment for excessive deviation
        self.R_dp = -0.5

        # Drastic punishment for collision
        self.R_cp = -1.0

        # The drone's position the last time the rewards were evaluated
        self.last_position = None
    
    def f_delta_t(self, dt):
        """This is the function that regulates the discount rate based on dmax"""
        return 0.5 * ((torch.tanh((2*self.d_max - dt)/self.d_max)) + 1)
    
    def determine_reward(self, did_collide, position):
        """This function determines the reward for a given time step"""

        # If collision, return drastic punishment
        if did_collide:
            # print("Harshest punishment - collision")
            return self.R_cp

        # If excessive deviation, mild punishment
        if torch.abs(self.dt_end) > self.d_max:
            return self.R_dp

        # If stays at (roughly) same point, return minimal reward
        stay_reward = 0
        if self.last_position is not None:
            if torch.allclose(position, self.last_position, atol=0.25):
                return stay_reward
        self.last_position = position.clone()
        
        # Calculate delta_d
        self.delta_d = self.dt_end - self.dt_start
        # Calculate reward as given in paper
        if self.delta_d > self.delta_d_u:
            return self.R_l * self.f_delta_t(self.dt_end)# + stay_reward
        elif self.delta_d < self.delta_d_l:
            return self.R_u * self.f_delta_t(self.dt_end)# + stay_reward
        else:
            return (self.R_l +
                (self.R_u - self.R_l)*
                ((self.delta_d_u - self.delta_d)/(self.delta_d_u - self.delta_d_l))
            )*self.f_delta_t(self.dt_end)# + stay_reward

