#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from xml.dom import minidom
import random
import os
import numpy as np
from collections import deque


class MazeGenerator:
    def __init__(self, maze_size=3.0, num_obstacles=8, output_dir="mazes"):
        self.maze_size = maze_size
        self.num_obstacles = num_obstacles
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # RANDOM WALL GENERATION
    # ----------------------------
    def generate_random_wall(self):
        #choose random length between 0.3 and 1.0
        length = random.uniform(0.3, 1.0)

        #if horizontal, choose a random x value for the center and add half the length on both sides
        if random.choice(['h', 'v']) == 'h':
            y = random.uniform(-self.maze_size + 0.5, self.maze_size - 0.5)
            x_center = random.uniform(-self.maze_size + 0.5, self.maze_size - 0.5)
            x1 = x_center - length / 2
            x2 = x_center + length / 2
            y1 = y2 = y
        #if vertical, choose a random y value for the center and add half the length to both sides
        else:
            x = random.uniform(-self.maze_size + 0.5, self.maze_size - 0.5)
            y_center = random.uniform(-self.maze_size + 0.5, self.maze_size - 0.5)
            y1 = y_center - length / 2
            y2 = y_center + length / 2
            x1 = x2 = x

        return (x1, y1, x2, y2)

    
    #Goal randomization for DR
    def generate_random_goal(self, walls):
        #so robot is not too close to the wall
        margin = 0.3

        while True:
            x = random.uniform(-self.maze_size + margin, self.maze_size - margin)
            y = random.uniform(-self.maze_size + margin, self.maze_size - margin)

            if not self.is_near_wall((x, y), walls):
                return (x, y)

    #walls is a list of walls and each wall is a line segment
    def is_near_wall(self, point, walls, threshold=0.2):
        px, py = point

        for x1, y1, x2, y2 in walls:
            dx = x2 - x1
            dy = y2 - y1

            if dx == 0 and dy == 0:
                continue

            t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / (dx*dx + dy*dy)))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy

            dist = ((px - proj_x)**2 + (py - proj_y)**2)**0.5

            if dist < threshold:
                return True

        return False

    #convert continuous maze to discrete occupanct grid to run bfs
    #resolution = size of each grid cell
    def build_grid(self, walls, resolution=0.15):
        #total continuous size is from -maze.size to +maze.size, size  = total number of grid cells 
        size = int((2 * self.maze_size) / resolution)
        #create sizexsize grid and fill with zeros
        grid = np.zeros((size, size), dtype=int)

        #convert coordinates to grid indices
        def to_grid(x, y):
            #world is centered at (0,0), need to shift coordinates into positve space because arrays start at (0,0)
            gx = int((x + self.maze_size) / resolution)
            gy = int((y + self.maze_size) / resolution)
            return gx, gy
        
        #process each wall segment
        for x1, y1, x2, y2 in walls:
            steps = int(max(abs(x2 - x1), abs(y2 - y1)) / resolution * 2)

            for i in range(steps + 1):
                t = i / steps if steps > 0 else 0
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)

                gx, gy = to_grid(x, y)

                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            grid[nx, ny] = 1

        return grid, to_grid

    # ----------------------------
    # BFS REACHABILITY
    # ----------------------------
    def is_reachable(self, grid, to_grid, start, goal):
        sx, sy = to_grid(*start)
        gx, gy = to_grid(*goal)

        queue = deque([(sx, sy)])
        visited = set([(sx, sy)])

        directions = [(1,0), (-1,0), (0,1), (0,-1)]

        while queue:
            x, y = queue.popleft()

            if (x, y) == (gx, gy):
                return True

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if (0 <= nx < grid.shape[0] and
                    0 <= ny < grid.shape[1] and
                    grid[nx, ny] == 0 and
                    (nx, ny) not in visited):

                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return False

    # ----------------------------
    # VALID MAZE GENERATION LOOP
    # ----------------------------
    def create_valid_maze(self):
        while True:
            walls = []

            # boundaries
            boundaries = [
                (-self.maze_size, -self.maze_size, -self.maze_size, self.maze_size),
                (-self.maze_size, self.maze_size, self.maze_size, self.maze_size),
                (self.maze_size, -self.maze_size, self.maze_size, self.maze_size),
                (-self.maze_size, -self.maze_size, self.maze_size, -self.maze_size)
            ]
            walls.extend(boundaries)

            for _ in range(self.num_obstacles):
                walls.append(self.generate_random_wall())

            goal = self.generate_random_goal(walls)

            grid, to_grid = self.build_grid(walls)

            starts = [(-2.5,-2.5), (2.5,-2.5), (-2.5,2.5), (2.5,2.5)]

            if any(self.is_reachable(grid, to_grid, s, goal) for s in starts):
                return walls, goal

    # ----------------------------
    # XML CREATION
    # ----------------------------
    def create_maze_xml(self):
        world = ET.Element('world')

        walls, goal_pos = self.create_valid_maze()

        # Start positions
        start_positions = ET.SubElement(world, 'experimentStartPositions')
        starts = [
            (-2.5, -2.5, 1.57),
            (2.5, -2.5, 1.57),
            (-2.5, 2.5, 1.57),
            (2.5, 2.5, 1.57)
        ]

        for x, y, theta in starts:
            pos = ET.SubElement(start_positions, 'pos')
            pos.set('x', str(x))
            pos.set('y', str(y))
            pos.set('theta', str(theta))

        # Goal
        goal = ET.SubElement(world, 'goal')
        goal.set('id', '1')
        goal.set('x', str(goal_pos[0]))
        goal.set('y', str(goal_pos[1]))

        # Walls
        for x1, y1, x2, y2 in walls:
            wall = ET.SubElement(world, 'wall')
            wall.set('x1', str(x1))
            wall.set('y1', str(y1))
            wall.set('x2', str(x2))
            wall.set('y2', str(y2))

        return world

    def prettify_xml(self, elem):
        rough = ET.tostring(elem, 'utf-8')
        return minidom.parseString(rough).toprettyxml(indent="\t")

    def generate_multiple_mazes(self, count=5):
        for i in range(count):
            maze = self.create_maze_xml()
            xml = self.prettify_xml(maze)

            filename = os.path.join(self.output_dir, f"maze_{i}.xml")
            with open(filename, "w") as f:
                f.write(xml)

            print("Saved:", filename)


def main():
    output_path = r"c:\Users\Johan\OneDrive\Documents\School\FAIRIS\simulation\worlds\mazes\Experiment1"
    generator = MazeGenerator(output_dir=output_path)
    generator.generate_multiple_mazes(3)


if __name__ == "__main__":
    main()