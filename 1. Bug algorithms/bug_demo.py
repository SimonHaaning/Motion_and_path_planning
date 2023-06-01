# Demo for the bug algorithms
from Simulation.simulation_obstacles import bug_maze, tiago_obstacles
from shapely.geometry import Polygon, Point, LineString, MultiLineString
import math
import matplotlib.pyplot as plt

# Import maze setup
obstacle, start, goal = bug_maze()
# obstacle, start, goal = tiago_obstacles()


# bug 0
def bug0(start_pos, goal_pos, obstacles, step_size=0.001, iterations=10000):
    current_pos = start_pos
    path_list = []

    hit_wall = False
    while iterations > 0:
        if current_pos.distance(goal_pos) <= step_size:
            break  # Reached goal

        # Add point to list for plotting
        path_list.append(current_pos)

        if not hit_wall:
            # Move towards goal
            angle_to_goal = math.atan2(goal_pos.y - current_pos.y, goal_pos.x - current_pos.x)
            new_x = current_pos.x + math.cos(angle_to_goal) * step_size
            new_y = current_pos.y + math.sin(angle_to_goal) * step_size
            new_pos = Point(new_x, new_y)
        else:
            new_pos = obstacles.boundary.interpolate(obstacles.boundary.project(current_pos) + step_size)

            # Check if the point is near a corner
            if any(new_pos.distance(Point(vertex)) <= step_size for vertex in obstacles.exterior.coords):
                hit_wall = False

            # Check for collision
        if obstacles.contains(new_pos):
            # Move the point to the nearest point on the polygon's boundary
            new_pos = obstacles.boundary.interpolate(obstacles.boundary.project(new_pos))
            hit_wall = True

        current_pos = new_pos
        iterations -= 1
    return path_list


# Bug 1
def bug1(start_pos, goal_pos, obstacles, step_size=0.001, iterations=10000):
    current_pos = start_pos
    path_list = []

    hit_point = None
    obstacle_perimeter = []

    hit_wall = False
    while iterations > 0:
        if current_pos.distance(goal_pos) <= step_size:
            break  # Reached goal

        # Add point to list for plotting
        path_list.append(current_pos)

        if not hit_wall:
            # Move towards goal
            angle_to_goal = math.atan2(goal_pos.y - current_pos.y, goal_pos.x - current_pos.x)
            new_x = current_pos.x + math.cos(angle_to_goal) * step_size
            new_y = current_pos.y + math.sin(angle_to_goal) * step_size
            new_pos = Point(new_x, new_y)
        else:
            # circumnavigate
            new_pos = obstacles.boundary.interpolate(obstacles.boundary.project(current_pos) - step_size)

            # Check if the point is near a previously visited point
            if new_pos.distance(hit_point) < step_size*0.99:
                # Determine leave point
                shortest_distance = math.inf
                for point in obstacle_perimeter:
                    distance = point.distance(goal_pos)

                    if distance < shortest_distance:
                        shortest_distance = distance
                        closest_point = point

                new_pos = closest_point
                hit_wall = False
            else:
                obstacle_perimeter.append(new_pos)

            # Check for collision
        if obstacles.contains(new_pos):
            # Move the point to the nearest point on the polygon's boundary
            new_pos = obstacles.boundary.interpolate(obstacles.boundary.project(new_pos))
            hit_point = new_pos
            hit_wall = True

        current_pos = new_pos
        iterations -= 1
    return path_list


# Bug 2
def bug2(start_pos, goal_pos, obstacles, step_size=0.001, iterations=10000):
    current_pos = start_pos
    path_list = []

    hit_point = None
    leave_point = None
    obstacle_perimeter = []

    hit_wall = False
    while iterations > 0:
        if current_pos.distance(goal_pos) <= step_size:
            break  # Reached goal

        # Add point to list for plotting
        path_list.append(current_pos)

        if not hit_wall:
            # Move towards goal
            angle_to_goal = math.atan2(goal_pos.y - current_pos.y, goal_pos.x - current_pos.x)
            new_x = current_pos.x + math.cos(angle_to_goal) * step_size
            new_y = current_pos.y + math.sin(angle_to_goal) * step_size
            new_pos = Point(new_x, new_y)
        else:
            # Travel around until finding the leave point
            new_pos = obstacles.boundary.interpolate(obstacles.boundary.project(current_pos) - step_size)

            # Check if the point is near a previously visited point
            if new_pos.distance(Point(leave_point)) < step_size*0.99:
                hit_wall = False
            else:
                obstacle_perimeter.append(new_pos)

        # Check for collision
        if obstacles.contains(new_pos):
            # Move the point to the nearest point on the polygon's boundary
            new_pos = obstacles.boundary.interpolate(obstacles.boundary.project(new_pos))

            # Determine leave point
            m_line = LineString([new_pos, goal_pos])
            m_intersect = m_line.intersection(obstacles)
            if isinstance(m_intersect, MultiLineString):
                for line in m_intersect.geoms:
                    leave_point = line.coords[-1]
            else:
                leave_point = m_intersect.coords[-1]
            hit_wall = True

        current_pos = new_pos
        iterations -= 1
    return path_list


# Bug tangent

# Plot environment and shortest path
path = bug0(start, goal, obstacle)
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Bug0')
ax.add_patch(plt.Polygon(obstacle.exterior, color='gray'))
ax.plot(start.x, start.y, 'o', color='green')
ax.plot(goal.x, goal.y, 'o', color='red')
for pos in path:
    ax.plot(pos.x, pos.y, 'o', markersize=1, color='blue')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.show()

# Plot environment and shortest path
path = bug1(start, goal, obstacle)
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Bug1')
ax.add_patch(plt.Polygon(obstacle.exterior, color='gray'))
ax.plot(start.x, start.y, 'o', color='green')
ax.plot(goal.x, goal.y, 'o', color='red')
for pos in path:
    ax.plot(pos.x, pos.y, 'o', markersize=1, color='blue')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.show()

# Plot environment and shortest path
path = bug2(start, goal, obstacle)
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Bug2')
ax.add_patch(plt.Polygon(obstacle.exterior, color='gray'))
ax.plot(start.x, start.y, 'o', color='green')
ax.plot(goal.x, goal.y, 'o', color='red')
for pos in path:
    ax.plot(pos.x, pos.y, 'o', markersize=1, color='blue')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.show()
