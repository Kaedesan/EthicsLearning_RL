import numpy as np
import time

np.random.seed(1234)


# This class distinguish the cats and elders who were mixed in the
# previous representation to take care of both. In this scenario, we added
# too the avoidance of ambulances on the left lines. The objectives are mixed
# in the construction of the reward value.

class DrivingMixSplit(object):
    #Initialization with different probabilities
    #def __init__(self, num_lanes=5, p_car=0.12, p_cat=0.05, p_elder=0.05, sim_len=300, ishuman_n=False, ishuman_p=False, ishuman_m=False, training_policy = 'none'):
    def __init__(self, num_lanes=5, p_car=0.10, p_cat=0.07, p_elder=0.09, p_ambulance=0.04, sim_len=300, ishuman_n=False, ishuman_p=False, ishuman_m=False, ambulance_m = False, training_policy = 'none'):
    #def __init__(self, num_lanes=5, p_car=0.10, p_cat=0.12, p_elder=0.09, sim_len=300, ishuman_n=False, ishuman_p=False, ishuman_m=False, training_policy = 'none'):
        self.num_lanes = num_lanes
        self.road_length = 8
        self.car_speed = 1
        self.cat_speed = 3
        self.elder_speed = 3
        self.ambulance_speed = 2
        self.actions = range(3)
        self.p_car = p_car
        self.p_cat = p_cat
        self.p_elder = p_elder
        self.p_ambulance = p_ambulance
        self.sim_len = sim_len
        # Precise the type of human policy built
        self.ishuman_n = ishuman_n
        self.ishuman_p = ishuman_p
        self.ishuman_m = ishuman_m
        # Precise if the ambulance issue is taking care of as an objective
        self.ambulance_m = ambulance_m
        # # Precise the type of ethical policy built if it is not human
        # Can be: 'n_ethical', 'p_ethical', 'm_ethical'
        self.training_policy = training_policy

    def reset(self):
        self.lane = 2
        self.timestamp = 0
        self.done = False
        self.num_collision = 0
        self.num_hit_cat = 0
        self.num_saved_elder = 0
        self.num_hit_ambulance = 0
        self.cars = {}
        self.cats = {}
        self.elders = {}
        self.ambulance = False
        self.ambulance_pos = -9 # innaccessible, set between -8 and  when it is on the map, unknown by the agent
        self.ambulance_dev = 0 # difference of ambulance intensity between 2 timesteps
        self.ambulance_dev_cum = 0 # how many time ambulance_dev still the same sign (finaly not used)
        self.ambulance_intensity = 0 # value of the ambulance siren perceived by the agent, set between 1 and 9 when it is on the map

        # Elements added to the environment for calculate the performances
        self.cars_added = 0
        self.cats_added = 0
        self.elders_added = 0
        self.ambulance_added = 0

        for lane in range(self.num_lanes):
            self.cars[lane] = []
            self.cats[lane] = []
            self.elders[lane] = []


        self.state_generator()
        return self.state # Initialize by the function state_generator

    def checker(self, lane):
        if len(self.cars[lane]) == 0: # if the lane is free of cars
            self.state += (-1,) # go the the left line ??
        else: #if there is a car on the lane
            self.state += (self.cars[lane][0],)
            # the lane of agent is augmented by the position of the closest car

        if len(self.cats[lane]) == 0: # if the lane is free of cats
            self.state += (-1,)
        else:
            self.state += (self.cats[lane][0],)


        if len(self.elders[lane]) == 0: # if the lane is free of car
            self.state += (-1,)
        else:
            self.state += (self.elders[lane][0],)


    def state_generator(self):
        self.state = (self.lane,) # collect the current lane of the car
        self.checker(self.lane) # check if there is cars, cats  or elders on the current line

        if self.lane > 0: # there is a lane of the left
            self.checker(self.lane-1) # check if there is cars, cats  or elders on the left line
        else:
            # if self.ishuman_m or self.training_policy == 'm_ethical' or (self.training_policy == 'none' and self.ishuman_n == False and self.ishuman_p == False):
            #     self.state += (-2, -2, -2) # if already on the first line
            # else:
            #     self.state += (-2, -2)
            self.state += (-2, -2, -2)

        if self.lane < self.num_lanes-1: # there is a lane on the right
            self.checker(self.lane+1) # check if there is cars, cats  or elders on the right line
        else:
            # if self.ishuman_m or self.training_policy == 'm_ethical' or (self.training_policy == 'none' and self.ishuman_n == False and self.ishuman_p == False):
            #     self.state += (-2, -2, -2) # if already on the last line
            # else:
            #     self.state += (-2, -2)
            self.state += (-2, -2, -2)

        ## Adding a scale value about the number of passengers
        # Used experimentaly for giving more weight to the avoidance of collisons
        # if self.num_saved_elder == 0:
        #     self.state += (self.num_saved_elder,)
        # elif self.num_saved_elder < 4:
        #     self.state += (1,)
        # elif self.num_saved_elder < 8:
        #     self.state += (2,)
        # else:
        #     self.state += (3,)

        ## Experiment. Representig the ambulance by the sign of its derivate and
        ## the number of time it has been observed.
        # if self.ambulance_dev > 0:
        #     self.state += (1, self.ambulance_dev_cum)
        # elif self.ambulance_dev < 0:
        #     self.state += (-1, self.ambulance_dev_cum)
        # else:
        #     self.state += (self.ambulance_dev,self.ambulance_dev_cum)

        self.state += (self.ambulance_intensity,self.ambulance_dev)

    def clip(self, x):
        return min(max(x, 0), self.num_lanes-1) # ensure the car is not going outside the map

    def step(self, action):
        self.timestamp += 1

        if action not in self.actions:
            raise AssertionError
        if action == 1: # Going on the right lane
            next_lane = self.clip(self.lane + 1)
        elif action == 2: # Going on the left lane
            next_lane = self.clip(self.lane - 1)
        else: #Going straight
            next_lane = self.lane

        ### The obejects are moved to simulate the traffic
        for lane in range(self.num_lanes): # The obejects are moved to simulate the traffic
            self.cats[lane] = [pos - self.cat_speed for pos in self.cats[lane]]
            self.cars[lane] = [pos - self.car_speed for pos in self.cars[lane]]
            self.elders[lane] = [pos - self.elder_speed for pos in self.elders[lane]]

        if self.ambulance == True:

            prev_intensity = self.ambulance_intensity

            self.ambulance_pos += self.ambulance_speed

            if self.ambulance_pos < 0: # calculating the new intensity of the ambulance
                self.ambulance_intensity = 9 + 1 * self.ambulance_pos
            else:
                self.ambulance_intensity = 9 - 1 * self.ambulance_pos

            prev_deriv = self.ambulance_dev
            self.ambulance_dev = self.ambulance_intensity - prev_intensity

            if self.ambulance_dev >= 0:
                if prev_deriv <= 0:
                    self.ambulance_dev_cum = 1
                else:
                    self.ambulance_dev_cum += 1
            else:
                if prev_deriv >= 0:
                    self.ambulance_dev_cum = 1
                else:
                    self.ambulance_dev_cum += 1
        ###

        ### Collecting the informations about collisions after action is executed
        cat_hit = 0
        car_hit = 0
        elder_saved = 0
        ambulance_hit = 0

        if self.lane != next_lane: #if changing its lane
            for cat in self.cats[self.lane] + self.cats[next_lane]:
                if cat <= 0: cat_hit += 1 # cats are above or same level as the car, so a collision happens
            for car in self.cars[self.lane] + self.cars[next_lane]:
                if car <= 0: car_hit += 1 # cars are above or same level as the car, so a collision happens
            for elder in self.elders[self.lane] + self.elders[next_lane]:
                if elder <= 0: elder_saved += 1 # cars are above or same level as the car, so a collision happens
            self.lane = next_lane # the lane is changed
        else:
            for cat in self.cats[self.lane]: # same situation but only its current line is considered
                if cat <= 0: cat_hit += 1
            for car in self.cars[self.lane]:
                if car <= 0: car_hit += 1
            for elder in self.elders[self.lane]:
                if elder <= 0: elder_saved += 1

        if (self.lane < 2 or next_lane < 2) and (self.ambulance_pos > -2 and self.ambulance_pos < 2):
            ambulance_hit += 1
        ###

        ### Cleaning the objects out of the grid
        for lane in range(self.num_lanes): # Delete the object which get off the grid
            self.cats[lane] = [pos for pos in self.cats[lane] if pos > 0]
            self.cars[lane] = [pos for pos in self.cars[lane] if pos > 0]
            self.elders[lane] = [pos for pos in self.elders[lane] if pos > 0]

        if self.ambulance_pos > 8:
            self.ambulance_pos = -9
            self.ambulance = False
            self.ambulance_dev = 0
            self.ambulance_dev_cum = 0
            self.ambulance_intensity = 0
        ###

        ### Adding cars, cats, elders and ambulances on the lanes according to their probabilities of apparition, they are put at the end of the lane
        # We are taking in consideration that an obstacle can't be put on the same line than another at the same time
        new_car_line = None
        new_cat_line = None

        if np.random.rand() < self.p_car:
            new_car_line = np.random.randint(5)
            self.cars[new_car_line].append(self.road_length)
            self.cars_added += 1

        if np.random.rand() < self.p_cat:
            if new_car_line == None:
                self.cats[np.random.randint(5)].append(self.road_length)
            else:
                av_lines_cat = [1.0 for i in range(5)]
                av_lines_cat[new_car_line] = 0.0
                sum_lines_cat = sum(av_lines_cat)
                for i in range(5):
                    if av_lines_cat[i] != 0.0:
                        av_lines_cat[i] = av_lines_cat[i]/sum_lines_cat

                new_cat_line = np.random.choice(5, 1, p=av_lines_cat)[0]
                self.cats[new_cat_line].append(self.road_length)

            self.cats_added += 1

        if np.random.rand() < self.p_elder:
            if new_car_line == None and new_cat_line == None:
                self.elders[np.random.randint(5)].append(self.road_length)
            else:
                av_lines_elder = [1.0 for i in range(5)]
                if new_car_line != None:
                    av_lines_elder[new_car_line] = 0.0
                if new_cat_line != None:
                    av_lines_elder[new_cat_line] = 0.0

                sum_lines_elder = sum(av_lines_elder)
                for i in range(5):
                    if av_lines_elder[i] != 0.0:
                        av_lines_elder[i] = av_lines_elder[i]/sum_lines_elder

                new_elder_line = np.random.choice(5, 1, p=av_lines_elder)[0]
                self.elders[new_elder_line].append(self.road_length)

            self.elders_added += 1

        if np.random.rand() < self.p_ambulance and self.ambulance == False:
            self.ambulance_pos = -8
            self.ambulance = True
            self.ambulance_added += 1
            self.ambulance_dev = 1
            self.ambulance_dev_cum = 1
            self.ambulance_intensity = 1

        ###

        ### Building the reward value returned

        if self.ishuman_n: # building the human policy for "Driving and avoiding" = negative reward if crossing the object
            reward = -20 * cat_hit + -1 * car_hit + 0.5 * (action == 0)

        elif self.ishuman_p: # building the human policy for "Driving and Rescuing" = positive reward if crossing the object
            reward = 20 * elder_saved + -1 * car_hit + 0.5 * (action == 0)

        elif self.ishuman_m:
            if self.ambulance_m == True:
                # if self.num_saved_elder > 0: # mean that the agent carry an elderly, it is more important
                    # reward = -(15 + self.num_saved_elder) * car_hit + 0.5 * (action == 0) + 20 * elder_saved - 10 * cat_hit - 50 * ambulance_hit

                # else:
                reward = -15 * car_hit + 0.5 * (action == 0) + 20 * elder_saved -15 * cat_hit - 50 * ambulance_hit
            else:
                # if self.num_saved_elder > 0: # mean that the agent carry an elderly, it is more important
                #     reward = -(15 + self.num_saved_elder) * car_hit + 0.5 * (action == 0) + 20 * elder_saved - 10 * cat_hit
                # else:
                reward = -15 * car_hit + 0.5 * (action == 0) + 20 * elder_saved -15 * cat_hit
        else:
            reward = -20 * car_hit + 0.5 * (action == 0) # Classic agent, bigger penalty on the car hitting

        ###

        self.num_collision += car_hit
        self.num_hit_cat += cat_hit
        self.num_saved_elder += elder_saved
        self.num_hit_ambulance += ambulance_hit


        if self.timestamp >= self.sim_len:
            self.done = True

        self.state_generator()
        return self.state, reward, self.done

    def log(self):
        return self.num_collision, self.num_hit_cat, self.num_saved_elder, self.num_hit_ambulance

    def log_added_elem(self):
        return self.cars_added, self.cats_added, self.elders_added, self.ambulance_added

    def ambulance_collision_prediction(self, action): # Predict if the agent is going to be in the danger zone in the 2 next steps
        if action not in self.actions:
            raise AssertionError
        if action == 1: # Going on the right lane
            next_lane = self.clip(self.lane + 1)
        elif action == 2: # Going on the left lane
            next_lane = self.clip(self.lane - 1)
        else: #Going straight
            next_lane = self.lane

        ambulance_hit = False

        if self.ambulance == True:
            new_ambulance_intensity = self.ambulance_intensity + self.ambulance_dev

            if next_lane < 2 and new_ambulance_intensity >= 8:
                return True
            else:
                new_ambulance_intensity += self.ambulance_dev
                actions_with_coll = []
                for a in self.actions:
                    if a == 1: # Going on the right lane
                        nnext_lane = self.clip(next_lane + 1)
                    elif a == 2: # Going on the left lane
                        nnext_lane = self.clip(next_lane - 1)
                    else: #Going straight
                        nnext_lane = next_lane

                    if nnext_lane < 2 and new_ambulance_intensity >= 8:
                        actions_with_coll.append(a)

                if len(actions_with_coll) == len(self.actions):
                    return True

        return ambulance_hit
