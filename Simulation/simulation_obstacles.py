from shapely.geometry import Polygon, Point


def bug_maze():
    obstacle = Polygon([(0.10, 0.10),
                    (0.10, 0.90),
                    (0.80, 0.90),
                    (0.80, 0.20),
                    (0.70, 0.20),
                    (0.70, 0.80),
                    (0.20, 0.80),
                    (0.20, 0.20),
                    (0.40, 0.20),
                    (0.40, 0.50),
                    (0.50, 0.50),
                    (0.50, 0.10)])
    start = Point(0.30, 0.30)
    goal = Point(0.90, 0.40)
    return obstacle, start, goal


def tiago_obstacles():
    obstacle = Polygon([(0.25, 0.25),
                        (0.25, 0.30),
                        (0.475, 0.30),
                        (0.475, 0.75),
                        (0.525, 0.75),
                        (0.525, 0.30),
                        (0.75, 0.30),
                        (0.75, 0.25)])
    start = Point(0.51, 0.10)
    goal = Point(0.50, 0.90)
    return obstacle, start, goal


def tiago_obstacles_bounded():
    obstacle = [Polygon([(0.25, 0.25),
                        (0.25, 0.30),
                        (0.475, 0.30),
                        (0.475, 0.75),
                        (0.525, 0.75),
                        (0.525, 0.30),
                        (0.75, 0.30),
                        (0.75, 0.25)]),
                Polygon([(0.01, 0.01),
                         (0.01, 0.99),
                         (0.99, 0.99),
                         (0.99, 0.01),
                         (0.00, 0.01),
                         (1.00, 0.00),
                         (1.00, 1.00),
                         (0.00, 1.00),
                         (0.00, 0.00)])]
    start = Point(0.51, 0.10)
    goal = Point(0.50, 0.90)
    return obstacle, start, goal


def ur5_obstacles():
    obs1 = Polygon([(-1.5, -1.5), (-1.5, -0.5), (-0.5, -0.5), (-0.5, -1.5)])
    obs2 = Polygon([(-1.0, 1.0), (-1.0, 2.0), (0.0, 2.0), (0.0, 1.0)])
    obs3 = Polygon([(0.5, -1.1), (0.5, -0.1), (1.5, -0.1), (1.5, -1.1)])
    obstacles = [obs1, obs2, obs3]
    return obstacles
