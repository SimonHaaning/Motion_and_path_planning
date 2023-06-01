import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Define the size of the maze
maze_width = 50
maze_height = 50

# Define the obstacle map
obstacle_map = np.zeros((maze_height, maze_width))  # Initialize with all zeros
obstacle_map[1:-1, 1:-1] = 0.6

# Set obstacles in the maze
obstacle_map[10:15, 20:30] = 0
obstacle_map[30:35, 10:20] = 0
obstacle_map[20:40, 40:45] = 0

# Define RRT parameters
max_iter = 1000  # Maximum number of iterations
step_size = 3  # Step size for growing the tree

# Define the RRT tree
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

def rrt(maze, start, goal, max_iter, step_size):
    # Initialize the RRT tree
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    nodes = [start_node]

    # RRT algorithm
    for i in range(max_iter):
        # Randomly sample a point
        x = np.random.randint(0, maze_width)
        y = np.random.randint(0, maze_height)
        rand_node = Node(x, y)

        # Find the nearest node in the tree
        nearest_node = None
        min_dist = float('inf')
        for node in nodes:
            dist = np.sqrt((node.x - rand_node.x)**2 + (node.y - rand_node.y)**2)
            if dist < min_dist:
                nearest_node = node
                min_dist = dist

        # Extend the tree towards the sampled point
        if min_dist > step_size:
            theta = np.arctan2(rand_node.y - nearest_node.y, rand_node.x - nearest_node.x)
            new_node = Node(int(nearest_node.x + step_size * np.cos(theta)),
                            int(nearest_node.y + step_size * np.sin(theta)))
        else:
            new_node = rand_node

        # Check for collision with obstacles
        collision = False
        x_vals = np.linspace(nearest_node.x, new_node.x, num=50, dtype=int)
        y_vals = np.linspace(nearest_node.y, new_node.y, num=50, dtype=int)
        if np.any(maze[y_vals, x_vals] == 0):
            collision = True

        if not collision:
            new_node.parent = nearest_node
            nodes.append(new_node)

            # Check if the goal is reached
            dist_to_goal = np.sqrt((new_node.x - goal_node.x)**2 + (new_node.y - goal_node.y)**2)
            if dist_to_goal <= step_size:
                goal_node.parent = new_node
                nodes.append(goal_node)
                return nodes

    return None

# Define the start and goal positions
start_pos = (5, 5)
goal_pos = (45, 45)
obstacle_map[start_pos] = 0.9
obstacle_map[goal_pos] = 0.9

# Run the RRT algorithm
path = rrt(obstacle_map, start_pos, goal_pos, max_iter, step_size)

# Prepare the line segments for visualization
lines = []
colors = []
for node_index, node in enumerate(path):
    if node.parent is not None:
        lines.append([(node.x, node.y), (node.parent.x, node.parent.y)])

# Calculate the colors using linear interpolation
num_colors = len(lines)
colors = plt.cm.jet(np.linspace(0, 1, num_colors))

# Visualize the maze and the RRT tree
fig, ax = plt.subplots()
ax.imshow(obstacle_map, cmap='gray')

# Create a LineCollection with the line segments and colors
lc = LineCollection(lines, colors=colors)
ax.add_collection(lc)

ax.plot(start_pos[0], start_pos[1], 'ro')
ax.plot(goal_pos[0], goal_pos[1], 'go')
ax.set_title('RRT Path Planning')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0, maze_width - 1)
ax.set_ylim(maze_height - 1, 0)
plt.show()
