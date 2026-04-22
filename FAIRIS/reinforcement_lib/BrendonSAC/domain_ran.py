#DOMAIN RANDOMIZATION METHODS LIST#
"""
Obstacle Randomization: Maze generator
LiDAR Noise: Randomly generate standard deviation in gaussian distribution
Linear Acceleration: Gaussian distribution for disatnce traveled
Actions: Gaussian distribution for angle that actions are taken
X and Y Values: Randomly generate standard deviation in gaussian distribution
Goal Location?:  Change goal location in maze generator
"""

# ==================== Domain Randomization Functions ====================

import numpy as np
 
def lidar_noise(lidar_readings, sigma_range=(0.005,0.05)):
    noisy_readings = []
    for reading in lidar_readings:
        sigma = np.random.uniform(*sigma_range)#pointer for the sigma_range (no longer hardcoded)
        noise = np.random.normal(0, sigma)
        noisy_readings.append(max(0, reading + noise))
    return np.array(noisy_readings)
 
 
def linear_acc_noise(linear_accel, sigma_range=(0.01, 0.1), clip_range=(-5.0, 5.0)):
   
    sigma = np.random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, size=np.shape(linear_accel))
    return np.clip(np.array(linear_accel) + noise, *clip_range)
 
 
# Number of discrete actions and their corresponding headings (degrees)
N_ACTIONS = 8
ACTION_ANGLES_DEG = np.array([i * (360 / N_ACTIONS) for i in range(N_ACTIONS)])  # [0, 45, 90, ..., 315]
 
 
def action_noise(action, sigma_range=(2.5, 15.0)):
   
    sigma_deg = np.random.uniform(*sigma_range)
    intended_angle_deg = ACTION_ANGLES_DEG[action]
    perturbed_angle_deg = intended_angle_deg + np.random.normal(0, sigma_deg)
 
    # Wrap to [0, 360) and find closest discrete action
    perturbed_angle_deg = perturbed_angle_deg % 360
    deltas = np.abs(ACTION_ANGLES_DEG - perturbed_angle_deg)
    # Handle wrap-around (e.g. distance between 350 deg and 10 deg)
    deltas = np.minimum(deltas, 360 - deltas)
    return int(np.argmin(deltas))

def pos_noise(x, y, sigma_range = (0.01,0.05)):
    """Add Gaussian noise to observed robot X/Y position."""
    sigma = np.random.uniform(*sigma_range)
    noisy_x = x + np.random.normal(0, sigma)
    noisy_y = y + np.random.normal(0, sigma)
    return float(noisy_x), float(noisy_y)

 
# =========================================================================

#MAZE GENERATOR (OBSTACLE RANDOMIZATION)#
#GOAL LOCATION - my understanding is we can use maze generator for this as well but 
#change the goal for the runs. Adding another layer. 
"""
Simple Random Maze Generator for Webots
Generates XML files compatible with the maze format shown in the example
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import random
import argparse
import os

class MazeGenerator:
    def __init__(self, maze_size=3.0, num_obstacles=8, output_dir="mazes"):
        """
        Initialize maze generator
        
        Args:
            maze_size: Size of the square maze (from -size to size)
            num_obstacles: Number of wall obstacles to generate
            output_dir: Directory to save XML files
        """
        self.maze_size = maze_size
        self.num_obstacles = num_obstacles
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_random_wall(self):
        """
        Generate a random wall obstacle
        Returns tuple of (x1, y1, x2, y2)
        """
        # Random wall length between 0.3 and 1.0
        length = random.uniform(0.3, 1.0)
        
        # Random orientation (horizontal or vertical)
        if random.choice(['h', 'v']) == 'h':
            # Horizontal wall
            y = random.uniform(-self.maze_size + 0.5, self.maze_size - 0.5)
            x_center = random.uniform(-self.maze_size + 0.5, self.maze_size - 0.5)
            x1 = x_center - length/2
            x2 = x_center + length/2
            y1 = y2 = y
        else:
            # Vertical wall
            x = random.uniform(-self.maze_size + 0.5, self.maze_size - 0.5)
            y_center = random.uniform(-self.maze_size + 0.5, self.maze_size - 0.5)
            y1 = y_center - length/2
            y2 = y_center + length/2
            x1 = x2 = x
            
        return (x1, y1, x2, y2)
    
    def create_maze_xml(self, maze_id=1,randomize_goal=False):
        """
        Create a complete maze XML structure
        """
        # Create root element
        world = ET.Element('world')
        
        # Add start positions (fixed positions from example)
        start_positions = ET.SubElement(world, 'experimentStartPositions')
        positions = [
            (-2.5, -2.5, 1.5707963267948966),
            (2.5, -2.5, 1.5707963267948966),
            (-2.5, 2.5, 1.5707963267948966),
            (2.5, 2.5, 1.5707963267948966)
        ]
        for x, y, theta in positions:
            pos = ET.SubElement(start_positions, 'pos')
            pos.set('x', str(x))
            pos.set('y', str(y))
            pos.set('theta', str(theta))
        
        # Add habituation start positions
        hab_start = ET.SubElement(world, 'habituationStartPositions')
        hab_positions = [
            (0.0, 2.5, 1.5707963267948966),
            (0.0, -2.5, 1.5707963267948966),
            (-2.5, 0.0, 1.5707963267948966),
            (2.5, 0.0, 1.5707963267948966)
        ]
        for x, y, theta in hab_positions:
            pos = ET.SubElement(hab_start, 'pos')
            pos.set('x', str(x))
            pos.set('y', str(y))
            pos.set('theta', str(theta))
        
        # Add goal (fixed at center)
        # goal = ET.SubElement(world, 'goal')
        # goal.set('id', '1')
        # goal.set('x', '0.0')
        # goal.set('y', '0.0')

        # Add goal
        goal = ET.SubElement(world, 'goal')
        goal.set('id', '1')
        if randomize_goal:
            # Keep goal away from corners (start positions) and boundaries
            goal_x = random.uniform(-self.maze_size * 0.5, self.maze_size * 0.5)
            goal_y = random.uniform(-self.maze_size * 0.5, self.maze_size * 0.5)
        else:
            goal_x, goal_y = 0.0, 0.0
        goal.set('x', str(round(goal_x, 4)))
        goal.set('y', str(round(goal_y, 4)))

        
        # Add outer walls (fixed boundaries)
        boundaries = [
            (-self.maze_size, -self.maze_size, -self.maze_size, self.maze_size),  # left
            (-self.maze_size, self.maze_size, self.maze_size, self.maze_size),   # top
            (self.maze_size, -self.maze_size, self.maze_size, self.maze_size),   # right
            (-self.maze_size, -self.maze_size, self.maze_size, -self.maze_size)  # bottom
        ]
        
        for x1, y1, x2, y2 in boundaries:
            wall = ET.SubElement(world, 'wall')
            wall.set('x1', str(x1))
            wall.set('y1', str(y1))
            wall.set('x2', str(x2))
            wall.set('y2', str(y2))
        
        # Add random obstacles
        for i in range(self.num_obstacles):
            x1, y1, x2, y2 = self.generate_random_wall()
            wall = ET.SubElement(world, 'wall')
            wall.set('x1', str(x1))
            wall.set('y1', str(y1))
            wall.set('x2', str(x2))
            wall.set('y2', str(y2))
        
        return world
    
    def prettify_xml(self, elem):
        """
        Return a pretty-printed XML string
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent='\t')
    
    def save_maze(self, filename, xml_content):
        """
        Save maze to XML file
        """
        with open(filename, 'w') as f:
            f.write(xml_content)
        print(f"Saved maze to: {filename}")
    
    def generate_multiple_mazes(self, count=10):
        """
        Generate multiple random mazes
        
        Args:
            count: Number of mazes to generate
        """
        for i in range(count):
            # Vary number of obstacles for domain randomization
            if random.random() < 0.3:  # 30% chance to vary obstacle count
                num_obstacles = random.randint(5, 12)
                self.num_obstacles = num_obstacles
            
            maze = self.create_maze_xml(i+1)
            xml_string = self.prettify_xml(maze)
            
            filename = os.path.join(self.output_dir, f'maze_{i+1:03d}.xml')
            self.save_maze(filename, xml_string)
        
        print(f"\nGenerated {count} random mazes in '{self.output_dir}/'")
    

    def generate_temp_maze(self,output_path, randomize_goal=False):
        #Generate a single randomized maze XML and save it to output_path."""
        if random.random() < 0.3:
            self.num_obstacles = random.randint(5, 12)
        maze = self.create_maze_xml(randomize_goal=randomize_goal)
        xml_string = self.prettify_xml(maze)
        with open(output_path, 'w') as f:
            f.write(xml_string)
        return output_path


def main():
    output_path = 'C:\\Users\\ploop\\Documents\\CIS4915\\FAIRIS\\simulation\\worlds\\mazes\\Experiment1'
    
    
    
    # Create generator
    generator = MazeGenerator(
        maze_size=3.0,
        num_obstacles=8,
        output_dir=output_path
    )
    
    # Generate mazes
    generator.generate_multiple_mazes(2)
    
    
if __name__ == "__main__":
    main()

# =========================================================================
2