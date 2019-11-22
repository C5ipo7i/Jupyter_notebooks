import numpy as np 
import copy
import random
from functools import reduce

class OUnoise(object):
    def __init__(self,size,seed,mu=0,theta=0.15,sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        self.length = reduce((lambda x,y: x*y),size)

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.length).reshape(self.size)
        self.state = x+dx
        return self.state