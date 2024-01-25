import gymnasium as gym
#from gym import spaces
import pygame
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from copy import deepcopy



# Pandas config
pd.set_option("display.max_rows", None, "display.max_columns", None)

# ****************************** VNF Generator GLOBALS ******************************
# File directory
DIRECTORY = 'gym-examples/gym_examples/slice_request_db1'
#DIRECTORY = 'data/scripts/DQN_models/Model1/gym_examples/slice_request_db1' #For pod

# Number of VNF types dictionary
# i.e. {key: value}, value = [BW] (modified to 5 types of [BW] only)
VNF_TYPES = {0: [8], 1: [20], 2: [14], 3: [5], 4: [30], 5: [2]}
# Arrival rates from VNF types dictionary
ARRIVAL_RATE = {0: 3, 1: 2, 2: 3, 3: 4, 4: 2, 5: 3}
# VNF life cycle from VNF types dictionary
LIFE_CYCLE_RATE = {0: 10, 1: 8, 2: 5, 3: 3, 4: 9, 5: 10}
# Num of vnf requests
NUM_VNF_REQUESTS = 1000

# ****************************** VNF Generator FUNCTIONS ******************************

def generate_requests_per_type(key, num):
    """ This function generates a set of requests per type """
    req = []
    vnf_request_at_time = 0

    x = 0  # to check the inter arrival times
    y = 0  # to check the holding times

    for _ in range(num):
        # Generate inter-arrival time for the VNF request
        inter_vnf_time_request = np.random.exponential(1.0 / ARRIVAL_RATE[key])
        # Update the time for the next request
        vnf_request_at_time += inter_vnf_time_request

        # Generate holding time for the VNF request
        #vnf_request_life_time = np.random.exponential(LIFE_CYCLE_RATE[key])
        # Alternative: Use a Poisson distribution for holding time
        vnf_request_life_time = np.random.poisson(LIFE_CYCLE_RATE[key]) 
        vnf_kill_at_time = vnf_request_at_time + vnf_request_life_time

        final_vnf = [vnf_request_at_time, VNF_TYPES[key][0], vnf_kill_at_time]
        #final_vnf = [vnf_request_at_time, VNF_TYPES[key][0], vnf_kill_at_time]

        # Round up decimals
        final_vnf = [round(val, 3) if isinstance(val, (int, float)) else val for val in final_vnf]
        req.append(final_vnf)

        x += inter_vnf_time_request
        y += vnf_request_life_time

    # print("DEBUG: key = ", key, "average inter-arrival = ", x / num, "average holding = ", y / num)
    return req


def get_key(val):
    """ Get value key """
    for k, v in VNF_TYPES.items():
        if val == v:
            return k

def generate_vnf_list():
    # ****************************** MAIN CODE ******************************
    # The overall procedure to create the requests is as follows:
        # - generate a set of requests per type
        # - put them altogether
        # - sort them according the arrival time
        # - return the num_VNFs_requests number of them '''

    vnfList = []

    for vnf in list(VNF_TYPES.values()):
        # Get vnf key for the arrival and holding dicts
        key = get_key(vnf)

            # We don't know how many requests from each type will be in the final list of requests.
            # It depends on the arrival rate of the type in comparison to the rates of the other types.
            # So, we generate the maximum number, i.e., num_VNFs_requests.
        requests = generate_requests_per_type(key, NUM_VNF_REQUESTS)

            # vnfList will be all the requests from all types not sorted.
        for req in requests:
            vnfList.append(req)

    # Sort the requests according to the arrival rate
    vnfList.sort(key=lambda x: x[0])

        # Until now, we have generated num_VNFs_requests * len(vnf_types) requests.
        # We only need the num_VNFs_requests of them'''
    vnfList = vnfList[:NUM_VNF_REQUESTS]

        # Dataframe
    columns = ['ARRIVAL_REQUEST_@TIME', 'SLICE_BW_REQUEST', 'SLICE_KILL_@TIME']
    df = pd.DataFrame(data=vnfList, columns=columns, dtype=float)

        # Export df to  csv file
    df.to_csv(DIRECTORY, index=False, header=True)

#------------------------------------Environment Class----------------------------------------------------
class SliceCreationEnv1(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        # Define environment parameters
        
        #Available resources (Order: MEC BW, )
        self.resources = [1000]
        
        #Defined parameters per Slice. (Each component is a list of the correspondent slice parameters)
        self.slices_param = [10, 20, 50]

        self.slice_requests = pd.read_csv('/home/mario/Documents/DQN_Models/Model 1/gym-examples/gym_examples/slice_request_db1')  # Load VNF requests from the generated CSV
        #self.slice_requests = pd.read_csv('/data/scripts/DQN_models/Model1/gym_examples/slice_request_db1')    #For pod
        
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(2,), dtype=np.float32) #ovservation space composed by Requested resources (MEC BW) and available MEC resources.
        
        self.action_space = gym.spaces.Discrete(4)  # 0: Do Nothing, 1: Allocate Slice 1, 2: Allocate Slice 2, 3: Allocate Slice 3

        #self.process_requests()
        
        # Define other necessary variables and data structures
        self.current_time_step = 1
        self.reward = 0
        self.first = True
        self.resources_flag = 1
        
        self.processed_requests = []

    #-------------------------Reset Method----------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        # Initialize the environment to its initial state

        #print(self.processed_requests)

        generate_vnf_list()

        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.current_time_step = 1
        self.reward = 0
        self.processed_requests = []
        self.reset_resources()
        self.slice_requests = pd.read_csv('/home/mario/Documents/DQN_Models/Model 1/gym-examples/gym_examples/slice_request_db1')  # Load VNF requests from the generated CSV
        #self.slice_requests = pd.read_csv('/data/scripts/DQN_models/Model1/gym_examples/slice_request_db1')    #For pod
        self.next_request = self.read_request()
        self.update_slice_requests(self.next_request)
        self.check_resources(self.next_request[1])
        self.observation = np.array([self.next_request[1]] + [self.resources_flag], dtype=np.float32)
        #self.observation = np.array([self.next_request[1]] + deepcopy(self.resources), dtype=np.float32)
        self.info = {}
        self.first = True
        
        print("\nReset: ", self.observation)
        
        return self.observation, self.info

    #-------------------------Step Method-------------------------------------------------------------------------
    def step(self, action):
        
        if self.first:
            self.next_request = self.processed_requests[0]
            self.first = False
        #else: 
            #next_request = self.read_request()
        #    self.update_slice_requests(self.next_request)
            
        terminated = False
        
        slice_id = self.create_slice(self.next_request)
        
        reward_value = 1
        
        # Apply the selected action (0: Do Nothing, 1: Allocate Slice 1, 2: Allocate Slice 2, 3: Allocate Slice 3)
        terminated = self.evaluate_action(action, slice_id, reward_value, terminated) 

        self.update_slice_requests(self.next_request)

        self.check_resources(self.next_request[1])
    
        self.observation = np.array([self.next_request[1]] + [self.resources_flag], dtype=np.float32)
        
        #done = False
        
        info = {}  # Additional information (if needed)
        
        #self.current_time_step += 1  # Increment the time step
        
        #print("Action: ", action, "\nObservation: ", self.observation, "\nReward: ", self.reward)

        truncated = False
        
        return self.observation, self.reward, terminated, truncated, info
    
    def read_request(self):
        next_request = self.slice_requests.iloc[self.current_time_step - 1]
        request_list =list([next_request['ARRIVAL_REQUEST_@TIME'], next_request['SLICE_BW_REQUEST'], next_request['SLICE_KILL_@TIME']])
        self.current_time_step += 1
        return request_list
        
    def update_slice_requests(self, request):
        # Update the slice request list by deleting the killed VNFs
        if len(self.processed_requests) != 0:
            for i in self.processed_requests:
                #i[2] < request[0]
                if len(i)== 4 and i[2] <= request[0]:
                    self.deallocate_slice(i)
                    self.processed_requests.remove(i)
        self.processed_requests.append(request)
        
    def check_resources(self, slice_bw_request):
        # Logic to check if there are available resources to allocate the VNF request
        # Return True if resources are available, False otherwise
        if self.resources[0] >= int(slice_bw_request):
            self.resources_flag = 1
        else: self.resources_flag = 0
    
    def allocate_slice(self, slice_bw_request):
        # Allocate the resources requested by the current VNF
        self.resources[0] -= int(slice_bw_request)
        # Define Slice ID
    
    def deallocate_slice(self, request):
        # Function to deallocate resources of killed requests
        self.resources[0] = self.resources[0] + request[1]
        
    def create_slice (self, request):
        # Function to create the slice for a specific request
        # This function inserts the defined slice to the request in the processed requests list
        
        resources = request[1]
        if resources <= self.slices_param[0]:
            slice_id = 1
        elif resources <= self.slices_param[1]:
            slice_id = 2
        elif resources <= self.slices_param[2]:
            slice_id = 3
        return slice_id

    def reset_resources(self):
        self.resources = [1000]
    
    def evaluate_action(self, action, slice_id, reward_value, terminated):
        if action == 1 and slice_id == 1:
            self.check_resources(self.next_request[1])
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request[1])
                self.processed_requests[len(self.processed_requests) - 1].append(slice_id)
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 1 and slice_id != 1:
            terminated = True
            self.reward = 0
            
        if action == 2 and slice_id == 2:
            self.check_resources(self.next_request[1])
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request[1])
                self.processed_requests[len(self.processed_requests) - 1].append(slice_id)
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 2 and slice_id != 2:
            terminated = True
            self.reward = 0
            
        if action == 3 and slice_id == 3:
            self.check_resources(self.next_request[1])
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request[1])
                self.processed_requests[len(self.processed_requests) - 1].append(slice_id)
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 3 and slice_id != 3:
            terminated = True
            self.reward = 0
            
        if action == 0:
            self.check_resources(self.next_request[1])
            if self.resources_flag == 0:
                self.reward += reward_value
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0        
        
        return terminated

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            
#a = SliceCreationEnv1()
#check_env(a)