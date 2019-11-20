import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.distribution as dist
import torch.distributions.categorical as Categorical
from functools import reduce
import numpy as np 

"""
PyTorch implementation of Actor Critic class for PPO. 
Combined torso with dual output 

2 Options:
use categorical for each slice
use softmax and torch.log for whole

Inputs:
Active routes
Historical routes (for novelty)
Current distance (minus stripped routes)
Can use the mask in the foward pass to auto restrict which techniques to suggest.
"""

class PPO_net(nn.Module):
    def __init__(self,nS,nA,seed,indicies,hidden_dims=(256,128)):
        super(PPO_net,self).__init__()
        self.nS = nS
        self.nA = nA
        self.seed = torch.manual_seed(seed)
        self.indicies = indicies
        self.hidden_dims = hidden_dims
        # TODO implement own batchnorm function
        # self.batch = manual_batchnorm()
        
        # Layers
        self.input_layer = nn.Linear(self.nS,hidden_dims[0])
        # self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        # self.hidden_batches = nn.ModuleList()
        for i in range(1,len(hidden_dims)):
            # hidden_batch = nn.BatchNorm1d(hidden_dims[i-1])
            hidden_layer = nn.Linear(hidden_dims[i-1],hidden_dims[i])
            self.hidden_layers.append(hidden_layer)
            # self.hidden_batches.append(hidden_batch)

        
        # Action outputs, we softmax over the various classes for 1 per class (can change this for multi class)
        self.actor_output = nn.Linear(hidden_dims[-1],nA)
        self.critic_output = nn.Linear(hidden_dims[-1],1)

    def forward(self,state):
        """
        Expects state to be a torch tensor

        Outputs Action,log_prob, entropy and (state,action) value
        """
        assert isinstance(state,torch.Tensor)
        x = F.relu(self.input_layer(state))
        for i,hidden_layer in enumerate(self.hidden_layers):
            x = F.relu(hidden_layer(x))
            
        # state -> action
        action = self.actor_output(x)
        a_prob = F.softmax(action,dim=0)
        log_prob = torch.log(action)
        # Critic state value
        v = self.critic_output(x)
        return a_prob, log_prob, v

"""
DDPG Network
"""

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def hard_update(source,target):
    for target_param,param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
    
class Critic(nn.Module):
    def __init__(self,seed,nS,nA,hidden_dims=(256,128)):
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_space = nA
        self.state_space = nS
        self.nS = [1]
        self.nA = reduce((lambda x,y: x*y),nA)
        [self.nS.append(x) for x in nS]
        self.nS = tuple(self.nS)
#         print('conv critic',hidden_dims[0],nA,nS)
        self.input_layer = nn.Conv1d(nS[0],hidden_dims[0],kernel_size=1)
        self.action_input = nn.Conv1d(nA[0],hidden_dims[0],kernel_size=1)
#         self.input_layer = nn.Linear(64,hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(nS[1]+nA[1],hidden_dims[0]))
        for i in range(0,len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        # self.fc1 = nn.Linear(hidden_dims[0]+nA,hidden_dims[1])
        # self.fc1_bn = nn.BatchNorm1d(hidden_dims[1])
        self.output_layer = nn.Linear(hidden_dims[-1]*hidden_dims[-2],1)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.output_layer.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,obs,action):
        N = obs.size()[0]
        # With batchnorm
#         obs = obs.view(-1)
#         action = action.view(-1)
        # xs = self.input_bn(F.relu(self.input_layer(state)))
        # x = torch.cat((xs,action),dim=1)
        # x = self.fc1_bn(F.relu(self.fc1(x)))
        # transpose obs and action
        action = F.relu(self.action_input(action))
        xs = F.relu(self.input_layer(obs))
        x = torch.cat((xs, action), dim=-1)
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.output_layer(x.view(N,-1))

class Actor(nn.Module):
    def __init__(self,seed,nS,nA,hidden_dims=(256,128)):
        super(Actor,self).__init__()
        print("nS,nA",nS,nA)
        self.seed = torch.manual_seed(seed)
        self.action_space = nA
        self.state_space = nS
        self.nS = [1]
        self.nA = reduce((lambda x,y: x*y),nA)
        [self.nS.append(x) for x in nS]
        self.nS = tuple(self.nS)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_layer = nn.Conv1d(nS[0],hidden_dims[0],kernel_size=1)
#         self.input_layer = nn.Linear(64,hidden_dims[0])
        self.fc1 = nn.Linear(nS[1],hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.output_layer = nn.Linear(hidden_dims[0]*hidden_dims[1],self.nA)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.output_layer.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state):
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32,device = self.device) #device = self.device,
            x = x.unsqueeze(0)
#         x = x.view(-1)
#         x = F.relu(self.conv1(x))
        N = state.size()[0]
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(N,-1)
        x = self.output_layer(x).view(N,self.action_space[0],self.action_space[-1])
        x = F.softmax(x,dim=-1)
        # print('networkoutput',x,x.shape,x.sum())
        return x