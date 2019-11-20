import numpy as np

class Hotel(object):
    def __init__(self,N_floors,N_elevators,N_people,capacity=None):
        """
        N_floors : Int, Number of floors for the building
        N_elevators : Int, Number of elevators
        N_people : Int, Number of people
        L_destinations : list, 
        Time_weights : list of weights, [0] weight for outside elevator, [1] weight for inside elevator
        """
        self.N_floors = N_floors
        self.N_elevators = N_elevators
        self.N_people = N_people
        
        # Constraints
        self.capacity = capacity
        self.time_weights = [1,1]
        self.reset()
        
    def step(self,action,printing=True):
        """
        count all time spent
        check if any elevators picked up or dropped people off
        return new state
        """
        self.move_elevators(action)
        self.check_elevators()
        reward = self.count_time()
        state = self.return_state()
        if printing == True:
            print(str(self))
        return state,reward,self.isdone
        
    def count_time(self):
        outside_mask = np.where(self.floors==1)[0]
        num_o_waiting = outside_mask.shape[0]
        num_o_waiting *= self.time_weights[0]
        # Inside elevators
        inside_mask = np.where(self.elevator_destinations == 1)[0]
        num_e_waiting = inside_mask.shape[0]
        num_e_waiting *= self.time_weights[1]
        self.cumulative_time += num_e_waiting + num_o_waiting
        return -(num_e_waiting + num_o_waiting)
    
    def move_elevators(self,action):
        self.elevator_targets = action
        # Move elevators 1 in the direction of the target
        # Get the vertical distance
        loc_mask = np.where(self.elevator_locations==1)[1]
        tar_mask = np.where(self.elevator_targets==1)[1]
        vert_distance = tar_mask - loc_mask
        elevators = np.arange(self.N_elevators)
        move = np.clip(vert_distance,-1,1)
        # print('move elevators',move,loc_mask)
        move_mask = loc_mask + move
        self.elevator_locations[elevators,loc_mask] = 0 
        self.elevator_locations[elevators,move_mask] = 1
        
    def check_elevators(self):
        loc_mask = np.where(self.elevator_locations==1)[1]
        tar_mask = np.where(self.elevator_targets==1)[1]
        vert_distance = tar_mask - loc_mask
        # If any target == the location, drop/pickup
        if 0 in vert_distance:
            # print('at location')
            stopped_mask = np.where(vert_distance == 0)[0]
            # print(self.elevator_destinations[stopped_mask])
            # print(self.elevator_locations[stopped_mask])
            # Check elevator destinations
            if 1 in self.elevator_destinations[stopped_mask]:
                for el in stopped_mask:
                    loc_mask = np.where(self.elevator_locations[el] == 1)
                    dest_mask = np.where(self.elevator_destinations[el] == 1)[0]
                    if loc_mask in dest_mask:
                        # Drop off
                        # print('passenger dropped off')
                        self.elevator_destinations[el,dest_mask] = 0
                        
                    
            # Check pickups
            if 1 in self.floors:
                # print('check pickups')
                for el in stopped_mask:
                    loc_mask = np.where(self.elevator_locations[el] == 1)[0] # elevator floor index
                    which_person = np.where(self.floors[:,loc_mask] == 1)[0] # Which person is at that floor (if any)
                    if which_person.size > 0:
                        # print('passenger picked up')
                        self.floors[which_person,:] = 0
                        self.elevator_destinations[[el]] += self.destinations[which_person]
                        self.destinations[which_person,:] = 0
                
    def return_state(self):
        # Concat together all floor and elevator states
        # print('elevator_locations.shape',self.elevator_locations.shape)
        # print('elevator_targets',self.elevator_targets.shape)
        # print('elevator_destinations',self.elevator_destinations.shape)
        machine_input = np.concatenate([self.floors,self.destinations,self.elevator_locations,self.elevator_targets,self.elevator_destinations],axis=0)
        # print('return state final',machine_input.shape)
        return np.expand_dims(machine_input, axis=0)
    
    def reset(self,top=True):
        # Floors
        self.floors = np.zeros((self.N_people,self.N_floors))
        self.destinations = np.zeros((self.N_people,self.N_floors))
        # Elevators
        self.elevator_locations = np.zeros((self.N_elevators,self.N_floors))
        self.elevator_targets = np.zeros((self.N_elevators,self.N_floors))
        self.elevator_destinations = np.zeros((self.N_elevators,self.N_floors))
        self.cumulative_time = 0
        # Generate floor destinations
        low = self.N_floors-1 if top == True else 0
        high = self.N_floors
        people_locations = np.random.choice(np.arange(self.N_floors),self.N_people,replace=False)
        people_destinations = np.random.randint(low,high,self.N_people)
        people = np.arange(self.N_people)
        self.floors[people,people_locations] = 1
        self.destinations[people,people_destinations] = 1
        # Generate elevator locations
        low_e = 0
        high_e = 1
        elevators = np.arange(self.N_elevators)
        elevator_locations = np.random.randint(low_e,high_e,self.N_elevators)
        self.elevator_locations[elevators,elevator_locations] = 1
        return self.return_state()
            
    @property
    def isdone(self):
        # Check if finish condition is met`
        outside_mask = np.where(self.floors==1)[0]
        num_o_waiting = outside_mask.shape[0]
        inside_mask = np.where(self.elevator_destinations == 1)[0]
        num_e_waiting = inside_mask.shape[0]
        if num_e_waiting + num_o_waiting == 0:
            finished = 1
        else:
            finished = 0
        return finished
    
    @property
    def action_space(self):
        return (self.N_elevators,self.N_floors)
    
    @property
    def state_space(self):
        return np.concatenate([self.floors,self.destinations,self.elevator_locations,self.elevator_targets,self.elevator_destinations],axis=0).shape
    
    # def reset(self):
    #     # Floors
    #     self.floors = np.zeros((self.N_people,self.N_floors))
    #     self.destinations = np.zeros((self.N_people,self.N_floors))
    #     # Elevators
    #     self.elevator_locations = np.zeros((self.N_elevators,self.N_floors))
    #     self.elevator_targets = np.zeros((self.N_elevators,self.N_floors))
    #     self.elevator_destinations = np.zeros((self.N_elevators,self.N_floors))
    #     self.cumulative_time = 0
    
    def __repr__(self):
        return '%s(%r)' % (self.__class__,self.__dict__)
    
    def __str__(self):
        elevator_locations = np.where(self.elevator_locations==1)[1]
        elevator_destinations = np.where(self.elevator_destinations==1)[1]
        people_locations = np.where(self.floors==1)[1]
        destinations = np.where(self.destinations==1)[1]
        return "elevator_locations {}, elevator_destinations {}, people_locations {}, destinations {}"    .format(elevator_locations,elevator_destinations,people_locations,destinations)