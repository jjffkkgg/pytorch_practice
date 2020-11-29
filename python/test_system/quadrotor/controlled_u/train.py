#%% In[1]

import quadrotor_A2C_ctrl as train_model
import numpy as np
import params as par
import json

#%% In[2]

if __name__ == '__main__':
    arrive_time = par.arrive_time
    hover_time = par.hover_time
    quadrotor_env = train_model.Environment()
    data = quadrotor_env.run(arrive_time, hover_time)
    np.save(par.datapath, data)
# %%
