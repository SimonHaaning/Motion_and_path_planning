import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import LineString
from Simulation.simulation_obstacles import ur5_obstacles

fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# Lengths of the robot arm segments
L1 = 0.9
L2 = 0.7

# Starting angles for the joints
theta1 = 0.0
theta2 = 0.0

# Scaling factors for animation
theta1_speed = 40
theta2_speed = 1/9

# Get obstacles
obstacles = ur5_obstacles()
for obs in obstacles:
    ax[0].add_patch(plt.Polygon(obs.exterior, color='black'))

collision_theta_values = []


# Function to detect collision
def check_collision(link, theta_1, theta_2):
    global obstacles, collision_theta_values
    line = LineString(link)
    for obstacle in obstacles:
        if obstacle.intersects(line):
            collision_theta_values.append((np.rad2deg(theta_1), np.rad2deg(theta_2)))
            return True
    return False


# Function to calculate the (x, y) coordinates of the end effector
def forward_kinematics(theta1, theta2):
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return x, y


# Initialize the robot arm plot
line1, = ax[0].plot([], [], 'b-', lw=5)  # Link 1
line2, = ax[0].plot([], [], 'b-', lw=5)  # Link 2
end_effector, = ax[0].plot([], [], 'bo', markersize=10)  # End effector


def init():
    # Set up the plot axes
    ax[0].set_xlim(-2, 2)
    ax[0].set_ylim(-2, 2)
    ax[0].set_aspect('equal')
    ax[1].set_xlim(0, 360)
    ax[1].set_ylim(0, 360)
    ax[1].set_aspect('equal')
    return line1, line2, end_effector


def update(frame):
    global collision_theta_values
    # Calculate the new joint angles
    theta1 = np.deg2rad(frame/theta1_speed) % (2 * np.pi)
    theta2 = np.deg2rad(frame/theta2_speed) % (2 * np.pi)

    # Calculate the (x, y) coordinates of the end effector
    x, y = forward_kinematics(theta1, theta2)

    # Update the plot data
    link1 = LineString([(0, 0), (L1 * np.cos(theta1), L1 * np.sin(theta1))])
    link2 = LineString([(L1 * np.cos(theta1), L1 * np.sin(theta1)), (x, y)])
    line1.set_data(*link1.xy)
    line2.set_data(*link2.xy)
    end_effector.set_data(x, y)
    collision_scatter = ax[1].scatter([], [], color='red')  # Collision points scatter plot

    # Change colors if colliding
    if check_collision(link1, theta1, theta2):
        line1.set_color('r')
    else:
        line1.set_color('b')
    if check_collision(link2, theta1, theta2):
        line2.set_color('r')
        end_effector.set_color('r')
    else:
        line2.set_color('b')
        end_effector.set_color('b')

    theta1_values = [theta1 for theta1, theta2 in collision_theta_values]
    theta2_values = [theta2 for theta1, theta2 in collision_theta_values]
    collision_scatter.set_offsets(np.column_stack((theta1_values, theta2_values)))

    return line1, line2, end_effector, collision_scatter

animation = animation.FuncAnimation(fig, update, frames=360*theta1_speed, init_func=init, blit=True, interval=0.1)

plt.show()