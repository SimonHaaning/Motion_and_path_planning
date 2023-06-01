import numpy as np
from shapely.geometry import Polygon, Point, LineString
from queue import PriorityQueue
import matplotlib.pyplot as plt
from scipy.stats import qmc


def generate_probabilistic_roadmap(num_nodes, max_distance, obstacles, start_pos, goal_pos, sampler='halton'):
    #Initialize sampler
    halton_sampler = qmc.Halton(d=2)
    halton = halton_sampler.random(num_nodes-2)
    uniform = np.random.rand(num_nodes-2, 2)

    if sampler == 'halton':
        # Generate list of nodes
        nodes = np.vstack([start_pos, goal_pos, halton])
    else:
        # Generate list of nodes
        nodes = np.vstack([start_pos, goal_pos, uniform])

    for i in range(len(nodes)):
        point = Point(nodes[i])
        for obs in obstacles:
            if obs.contains(point):
                nodes[i] = [None, None]
                break

    # Connect nodes within max_distance and without obstacles to form edges
    edges = []
    for i in range(len(nodes)):
        max_connections = 3 # The K parameter from the book (used to simplify the graph)
        for j in range(i + 1, len(nodes)):
            # Check if the distance between the nodes is less than max_distance the local controller can handle
            dist = np.linalg.norm(nodes[i] - nodes[j])

            if max_connections <= 0:
                break

            if dist <= max_distance:
                max_connections -= 1

                # Check if the edge between the nodes intersects any obstacles
                edge = LineString([nodes[i], nodes[j]])
                intersects = False
                for obs in obstacles:
                    if edge.intersects(obs):
                        intersects = True
                        break
                if not intersects:
                    edges.append((i, j))

    return nodes, edges


def dijkstra_shortest_path(nodes, edges, start_node_index, goal_node_index):
    # Initialize distance and predecessor dictionaries
    dist = {i: np.inf for i in range(len(nodes))}
    dist[start_node_index] = 0
    pred = {i: None for i in range(len(nodes))}

    # Initialize priority queue with start node
    pq = PriorityQueue()
    pq.put((0, start_node_index))

    # Dijkstra's algorithm
    while not pq.empty():
        curr_dist, curr_node_index = pq.get()

        if curr_node_index == goal_node_index:
            # Found shortest path, backtrack to construct path
            path = []
            curr = goal_node_index
            while curr is not None:
                path.append(curr)
                curr = pred[curr]
            path.reverse()
            return path

        for neighbor_node_index in get_neighbors(curr_node_index, edges):
            new_dist = dist[curr_node_index] + np.linalg.norm(nodes[curr_node_index] - nodes[neighbor_node_index])
            if new_dist < dist[neighbor_node_index]:
                dist[neighbor_node_index] = new_dist
                pred[neighbor_node_index] = curr_node_index
                pq.put((new_dist, neighbor_node_index))

    # No path found
    return None


def get_neighbors(node_index, edges):
    neighbors = []
    for edge in edges:
        if edge[0] == node_index:
            neighbors.append(edge[1])
        elif edge[1] == node_index:
            neighbors.append(edge[0])
    return neighbors


# Define obstacles as Shapely Polygons
obs1 = Polygon([(0.48, 0.2), (0.52, 0.2), (0.52, 0.8), (0.48, 0.8)])
obs2 = Polygon([(0.3, 0.2), (0.7, 0.2), (0.7, 0.24), (0.3, 0.24)])
obstacles = [obs1, obs2]

# Define start and goal positions
start_pos = (0.5, 0.9)
goal_pos = (0.5, 0.1)

# Generate probabilistic roadmap
nodes, edges = generate_probabilistic_roadmap(200, 0.3, obstacles, start_pos, goal_pos, 'halton')

# Find shortest path using Dijkstra's algorithm
start_node_index = 0
goal_node_index = 1
path = dijkstra_shortest_path(nodes, edges, start_node_index, goal_node_index)

# Plot environment and shortest path
fig, ax = plt.subplots()
for obs in obstacles:
    ax.add_patch(plt.Polygon(obs.exterior, color='gray'))
ax.plot(nodes[:, 0], nodes[:, 1], 'o', color='black', markersize=1)
for edge in edges:
    ax.plot(nodes[edge,0], nodes[edge,1], '--', color='blue', linewidth=0.1)
ax.plot(nodes[path,0], nodes[path,1], '-', color='red', linewidth=2)
ax.plot(start_pos[0], start_pos[1], 'o', color='green')
ax.plot(goal_pos[0], goal_pos[1], 'o', color='red')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.show()
