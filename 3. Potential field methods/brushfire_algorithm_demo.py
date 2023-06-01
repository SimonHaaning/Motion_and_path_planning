import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Each index is a cell
obstacles = np.zeros((100,100))

# Add border
obstacles[0, :] = 1
obstacles[-1, :] = 1
obstacles[:, 0] = 1
obstacles[:, -1] = 1

# Add obstacles
obstacles[30:80, 45:55] = 1
obstacles[20:30, 20:80] = 1

# Start and goal coordinates in the coordinate
start = (10, 50)
goal = (90, 50)


def brushfire_attractive(obstacles, goal):
    """
    Inputs:
        obstacles: a numpy array of shape (height, width) where 1s represent obstacles and 0s represent free space.
        start: a tuple (start_x, start_y) representing the starting position of the robot.
        goal: a tuple (goal_x, goal_y) representing the goal position of the robot.
    Returns:
        A numpy array of shape (height, width) representing the distance map, where each cell contains the minimum
        distance from that cell to the goal.
    """
    height, width = obstacles.shape
    distances = np.full_like(obstacles, height+width)
    visited = np.zeros_like(obstacles)
    queue = [goal]
    distances[goal] = 0  # set goal distance to 0

    while queue:
        current = queue.pop(0)
        visited[current] = 1
        neighbors = []
        x, y = current

        if x > 0:
            neighbors.append((x-1, y))
        if y > 0:
            neighbors.append((x, y-1))
        if x < height-1:
            neighbors.append((x+1, y))
        if y < width-1:
            neighbors.append((x, y+1))

        for neighbor in neighbors:
            if obstacles[neighbor] == 0 and not visited[neighbor]:
                new_distance = distances[current] + 1
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    queue.append(neighbor)

    return distances


def brushfire_repulsive(obstacles, stop_at_distance=10000):
    """
    Inputs:
        obstacles: a numpy array of shape (height, width) where 1s represent obstacles and 0s represent free space.
    Returns:
        A numpy array of shape (height, width) representing the distance map, where each cell contains the minimum
        distance from that cell to the nearest obstacle.
    """
    height, width = obstacles.shape
    distances = np.zeros_like(obstacles)

    # Forward pass
    for i in range(height):
        for j in range(width):
            if obstacles[i, j] == 0:
                min_distance = stop_at_distance
                if i > 0:
                    min_distance = min(min_distance, distances[i - 1, j] + 1)
                if j > 0:
                    min_distance = min(min_distance, distances[i, j - 1] + 1)
                distances[i, j] = min_distance

    # Backward pass
    for i in reversed(range(height)):
        for j in reversed(range(width)):
            if obstacles[i, j] == 0:
                if i < height-1:
                    distances[i, j] = min(distances[i, j], distances[i+1, j] + 1)
                if j < width-1:
                    distances[i, j] = min(distances[i, j], distances[i, j+1] + 1)

    return distances


# Attractive
attractive_distance = brushfire_attractive(obstacles, goal)
plt.imshow(attractive_distance, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.show()

# Repulsive
repulsive_distance = brushfire_repulsive(obstacles)
plt.imshow(repulsive_distance, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.show()

# Combined
combined_distance = attractive_distance + (repulsive_distance.max()-repulsive_distance)
plt.imshow(combined_distance, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.show()
