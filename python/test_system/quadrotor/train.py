# In[1]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.animation as animation
import quadrotor_A2C_traj as train_model
import numpy as np
import params as par

def update_lines(num, data, line):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        return line

# In[2]

if __name__ == '__main__':
    arrive_time = par.arrive_time
    hover_time = par.hover_time
    quadrotor_env = train_model.Environment()
    data, distance = quadrotor_env.run(arrive_time, hover_time)
    np.save('./python/test_system/quadrotor/trained_net/flight_data.npy', data)
    np.save('./python/test_system/quadrotor/trained_net/distance_data.npy', distance)
    
    t = np.arange(0,arrive_time + hover_time,par.DELTA_T)

    # Attaching 3D axis to the figure
    fig1 = plt.figure(num = 1, figsize=(plt.figaspect(1)))
    ax = Axes3D(fig1)

    x = data[0,:,9]
    y = data[0,:,10]
    z = data[0,:,11]
    data_plot = np.array([x,y,z])
    
    line = ax.plot(x, y, z)[0]

    # Setting the axes properties
    ax.set_xlim3d([-50, 50])
    ax.set_ylim3d([-50, 50])
    ax.set_zlim3d([0, 100])

    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.set_zlabel('Z')

    ax.set_title('Trajectory')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig1, update_lines, 25, fargs=(data_plot, line),
                                    interval=50, blit=False)
    line_ani.save('./python/test_system/quadrotor/trained_net/flight.gif', writer='pillow', fps=60)

    
    fig2 = plt.figure(num = 2, figsize=(16,9))
    ax1 = fig2.add_subplot(2, 3, 1)
    plt.plot(data[0,:,9],
                data[0,:,10])
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.title('x-y Postition')
    plt.xlabel('X axis [m]')
    plt.ylabel('Y axis [m]')
    plt.grid()
    
    ax2 = fig2.add_subplot(2, 3, 2)
    plt.plot(t,data[0,:,11])
    plt.title('Height')
    plt.xlabel('Time [s]')
    plt.ylabel('Height [m]')
    plt.grid()

    ax3 = fig2.add_subplot(2, 3, 3)
    plt.plot(t,data[0,:,6]*180/np.pi,
                t,data[0,:,7]*180/np.pi,
                t,data[0,:,8]*180/np.pi)
    plt.title('Angle')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [deg]')
    plt.legend(['x(phi)','y(theta)','z(psi)'])
    plt.grid()

    ax4 = fig2.add_subplot(2, 3, 4)
    plt.plot(t, data[0,:,0]*180/np.pi,
                t, data[0,:,1]*180/np.pi,
                t, data[0,:,2]*180/np.pi)
    plt.title('Angular Velocity(body)')
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity(body) [deg/s]')
    plt.legend(['x(phi)','y(theta)','z(psi)'])
    plt.grid()
    
    ax3 = fig2.add_subplot(2, 3, 5)
    plt.plot(t,distance[0,:])
    plt.title('Distance off from guided trajectory')
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.grid()

    fig2.savefig('./python/test_system/quadrotor/trained_net/flight_data.png')
    
    plt.show()