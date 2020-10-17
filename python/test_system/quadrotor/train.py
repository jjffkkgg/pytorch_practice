# In[1]

import quadrotor_A2C_traj as train_model
import numpy as np
import params as par

# In[2]

if __name__ == '__main__':
    arrive_time = par.arrive_time
    hover_time = par.hover_time
    quadrotor_env = train_model.Environment()
    data, distance, u_in = quadrotor_env.run(arrive_time, hover_time)
    np.save('./python/test_system/quadrotor/trained_net/flight_data.npy', data)
    np.save('./python/test_system/quadrotor/trained_net/distance_data.npy', distance)
    np.save('./python/test_system/quadrotor/trained_net/input_data.npy', u_in)
# %%
