import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

grid = np.array([
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# start point and goal
start = (0, 0)
goal = (0, 19)

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def edge_cost(a, b):
    return 0

def astar(array, start, goal, neighbors):
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    visited_set = []

    while oheap:
        current = heapq.heappop(oheap)[1]
        visited_set.append(current)
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data, visited_set

        close_set.add(current)

        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return False, visited_set


def djikstra(array, start, goal, neighbors):
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    visited_set = []

    while oheap:
        current = heapq.heappop(oheap)[1]
        visited_set.append(current)
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data, visited_set

        close_set.add(current)

        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + edge_cost(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return False, visited_set

neighbors8 = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
neighbors4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
routed, visitedd = djikstra(grid, start, goal, neighbors4)
routea, visiteda = astar(grid, start, goal, neighbors4)
routed = routed + [start]
routea = routea + [start]
routed = routed[::-1]
routea = routea[::-1]
print("\nDjikstra points: ")
print(len(list(set(visitedd))))
print("\nA* points: ")
print(len(list(set(visiteda))))


# extract x and y coordinates from route lists
x_coordsd = []
y_coordsd = []
x_coordsa = []
y_coordsa = []

for i in (range(0, len(routed))):
    x = routed[i][0]
    y = routed[i][1]
    x_coordsd.append(x)
    y_coordsd.append(y)
for i in (range(0, len(routea))):
    x = routea[i][0]
    y = routea[i][1]
    x_coordsa.append(x)
    y_coordsa.append(y)

# extract x and y coordinates from visited lists
x_coords_visitedd = []
y_coords_visitedd = []
x_coords_visiteda = []
y_coords_visiteda = []
for i in (range(0, len(visitedd))):
    x = visitedd[i][1]
    y = visitedd[i][0]
    x_coords_visitedd.append(x)
    y_coords_visitedd.append(y)
for i in (range(0, len(visiteda))):
    x = visiteda[i][1]
    y = visiteda[i][0]
    x_coords_visiteda.append(x)
    y_coords_visiteda.append(y)

# plot map and path
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(grid, cmap=plt.cm.Dark2)
ax.scatter(x_coords_visitedd, y_coords_visitedd, marker="o", color="white", s=200)
ax.scatter(x_coords_visiteda, y_coords_visiteda, marker="o", color="black", s=100)
ax.plot(y_coordsd, x_coordsd, color="white")
ax.plot(y_coordsa, x_coordsa, color="black")
ax.scatter(start[1], start[0], marker="*", color="red", s=200)
ax.scatter(goal[1], goal[0], marker="*", color="green", s=200)
plt.show()