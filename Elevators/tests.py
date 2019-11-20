from hotel import Hotel
import unittest

N_floors = 5
N_elevators = 1
N_people = 1

hotel = Hotel(N_floors,N_elevators,N_people)

#%% [markdown]
# ## New scenario

#%%
hotel.reset()
print(repr(hotel))
print(hotel)

#%% [markdown]
# ## Machine state

#%%
hotel.return_state().shape

#%% [markdown]
# ## Move elevators

#%%
test_action = np.zeros((N_elevators,N_floors))
test_action[0,4] = 1 # Move up
# test_action[0,0] = 1 # Move down
# 2 elevators
# test_action = np.zeros((N_elevators,N_floors))
# test_action[0,4] = 1 # Move up
# # test_action[1,2] = 1 # Move up
# test_action[0,4] = 1 # Move down
# test_action[1,4] = 1 # Move down
print('action',test_action)
hotel.move_elevators(test_action)


#%%
str(hotel)

#%% [markdown]
# ## Check elevators

#%%
hotel.check_elevators()

#%% [markdown]
# ## Step

#%%
hotel.reset()
str(hotel)
hotel.return_state()


#%%
test_action = np.zeros((N_elevators,N_floors))
test_action[0,4] = 1 # Move up
hotel.step(test_action)
