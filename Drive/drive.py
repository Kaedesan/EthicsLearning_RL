import numpy as np

np.random.seed(1234)

#Original class given by the paper

class Driving(object):
    def __init__(self, num_lanes=5, p_car=0.16, p_cat=0.09, sim_len=300, ishuman_n=False, ishuman_p=False):
        self.num_lanes = num_lanes
        self.road_length = 8
        self.car_speed = 1
        self.cat_speed = 3
        self.actions = range(3)
        self.p_car = p_car
        self.p_cat = p_cat
        self.sim_len = sim_len
        self.ishuman_n = ishuman_n
        self.ishuman_p = ishuman_p

    def reset(self):
        self.lane = 2
        self.timestamp = 0
        self.done = False
        self.num_collision = 0
        self.num_hit_cat = 0
        self.cars = {}
        self.cats = {}
        for lane in range(self.num_lanes):
            self.cars[lane] = []
            self.cats[lane] = []
        # the state shows the positions of the first cat and car in adjacent lanes
        self.state_generator()
        return self.state # Initialize by a function

    def checker(self, lane):
        if len(self.cars[lane]) == 0: # if the lane is free of cars
            self.state += (-1,) # go the the left line ??
        else: #if there is a car on the lane
            self.state += (self.cars[lane][0],)
            # the lane of agent is augmented by the position of the closest car
        if len(self.cats[lane]) == 0: # if the lane is free of car
            self.state += (-1,)
        else:
            self.state += (self.cats[lane][0],)

    def state_generator(self):
        self.state = (self.lane,) # collect the current lane of the car
        self.checker(self.lane) # check if there is cars or cats on the current line

        if self.lane > 0: # there is a lane of the left
            self.checker(self.lane-1) # check if there is cars or cats on the left line
        else:
            self.state += (-2, -2) # if already on the first line

        if self.lane < self.num_lanes-1: # there is a lane on the right
            self.checker(self.lane+1) # check if there is cars or cats on the right line
        else:
            self.state += (-2, -2) # if already on the last

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
        for lane in range(self.num_lanes): # The obejects are moved to simulate the traffic
            self.cats[lane] = [pos - self.cat_speed for pos in self.cats[lane]]
            self.cars[lane] = [pos - self.car_speed for pos in self.cars[lane]]

        cat_hit = 0
        car_hit = 0
        if self.lane != next_lane: #if changing its lane
            for cat in self.cats[self.lane] + self.cats[next_lane]:
                if cat <= 0: cat_hit += 1 # cats are above or same level as the car, so a collision happens
            for car in self.cars[self.lane] + self.cars[next_lane]:
                if car <= 0: car_hit += 1 # cars are above or same level as the car, so a collision happens
            self.lane = next_lane # the lane is changed
        else:
            for cat in self.cats[self.lane]: # same situation but only its current line is considered
                if cat <= 0: cat_hit += 1
            for car in self.cars[self.lane]:
                if car <= 0: car_hit += 1

        for lane in range(self.num_lanes): # Delete the object which get off the grid
            self.cats[lane] = [pos for pos in self.cats[lane] if pos > 0]
            self.cars[lane] = [pos for pos in self.cars[lane] if pos > 0]

        #Adding cars and cats on the lanes according to their probabilities of apparition, they are put at the end of the lane
        if np.random.rand() < self.p_car:
            self.cars[np.random.randint(5)].append(self.road_length)
        if np.random.rand() < self.p_cat:
            self.cats[np.random.randint(5)].append(self.road_length)

        if self.ishuman_n: # building the human policy for "Driving and avoiding" = negative reward if crossing the object
            reward = -20 * cat_hit + -1 * car_hit + 0.5 * (action == 0)
        elif self.ishuman_p: # building the human policy for "Driving and Rescuing" = positive reward if crossing the object
            reward = 20 * cat_hit + -1 * car_hit + 0.5 * (action == 0)
        else:
            reward = -20 * car_hit + 0.5 * (action == 0) # Classic agent, bigger penalty on the car hitting

        self.num_collision += car_hit
        self.num_hit_cat += cat_hit
        if self.timestamp >= self.sim_len:
            self.done = True

        self.state_generator()
        return self.state, reward, self.done

    def log(self):
        return self.num_collision, self.num_hit_cat
