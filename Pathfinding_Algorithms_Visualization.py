import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq
from time import time

def createGridworld(size, start, end):
    grid = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])
    grid[start] = 0
    grid[end] = 0
    return grid

def hValue(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def BFS(gridworld, start, end):
    compass = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    rows, cols = gridworld.shape
    queue = [start]
    visited = set([start])
    while queue:
        point = queue.pop(0)
        if point == end:
            return True
        for movement in compass:
            new_point = (point[0] + movement[0], point[1] + movement[1])
            if (0 <= new_point[0] < rows) and (0 <= new_point[1] < cols) and gridworld[new_point] == 0 and new_point not in visited:
                queue.append(new_point)
                visited.add(new_point)
    return False

def forwardALarge(grid, start, goal):
    compass = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    rows, cols = grid.shape
    tree = []
    heapq.heappush(tree, (0, start))
    g_values = {start: 0}
    path = {start: None}
    while tree:
        _, current = heapq.heappop(tree)
        if current == goal:
            break
        for dx, dy in compass:
            next_point = (current[0] + dx, current[1] + dy)
            if (0 <= next_point[0] < rows) and (0 <= next_point[1] < cols) and grid[next_point] == 0:
                new_g_value = g_values[current] + 1
                if next_point not in g_values or new_g_value < g_values[next_point]:
                    g_values[next_point] = new_g_value
                    f_value = new_g_value*0.9999 + hValue(goal, next_point) # 0.9999 bias
                    heapq.heappush(tree, (f_value, next_point))
                    path[next_point] = current
    return path

def forwardASmall(grid, start, goal):
    compass = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    rows, cols = grid.shape
    tree = []
    heapq.heappush(tree, (0, start))
    g_values = {start: 0}
    path = {start: None}
    while tree:
        _, current = heapq.heappop(tree)
        if current == goal:
            break
        for dx, dy in compass:
            next_point = (current[0] + dx, current[1] + dy)
            if (0 <= next_point[0] < rows) and (0 <= next_point[1] < cols) and grid[next_point] == 0:
                new_g_value = g_values[current] + 1
                if next_point not in g_values or new_g_value < g_values[next_point]:
                    g_values[next_point] = new_g_value
                    f_value = new_g_value*1.0001 + hValue(goal, next_point) # 1.0001 bias
                    heapq.heappush(tree, (f_value, next_point))
                    path[next_point] = current
    return path

def backwardA(grid, start, goal):
    compass = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    rows, cols = grid.shape
    tree = []
    heapq.heappush(tree, (0, start))
    g_values = {start: 0}
    path = {start: None}
    while tree:
        _, current = heapq.heappop(tree)
        if current == goal:
            break
        for dx, dy in compass:
            next_point = (current[0] + dx, current[1] + dy)
            if (0 <= next_point[0] < rows) and (0 <= next_point[1] < cols) and grid[next_point] == 0:
                new_g_value = g_values[current] + 1
                if next_point not in g_values or new_g_value < g_values[next_point]:
                    g_values[next_point] = new_g_value
                    f_value = new_g_value + hValue(goal, next_point)
                    heapq.heappush(tree, (f_value, next_point))
                    path[next_point] = current
    return path

def adaptiveA(grid, start, goal):
    compass = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    rows, cols = grid.shape
    tree = []
    heapq.heappush(tree, (0, start))
    g_values = {start: 0}
    path = {start: None}
    while tree:
        f_value, current = heapq.heappop(tree) # Get current point with lowest f_value
        if current == goal:
            break
        for dx, dy in compass:
            next_point = (current[0] + dx, current[1] + dy)
            if (0 <= next_point[0] < rows) and (0 <= next_point[1] < cols) and grid[next_point] == 0:
                new_g_value = g_values[current] + 1
                if next_point not in g_values or new_g_value < g_values[next_point]:
                    g_values[next_point] = new_g_value
                    f_value = new_g_value*0.9999 + hValue(goal, next_point) # 0.9999 bias
                    heapq.heappush(tree, (f_value, next_point))
                    path[next_point] = current
    return path

def animateForwardAL(grid, start, goal):
    fig_larger, ax_larger = plt.subplots()
    ax_larger.imshow(grid, cmap='binary', interpolation='nearest')
    ax_larger.set_title('Path Search Animation (Larger Bias)')
    ax_larger.set_xlabel('Columns')
    ax_larger.set_ylabel('Rows')
    path = forwardALarge(grid, start, goal)
    path_points = []
    current = goal
    while current is not None:
        path_points.append(current)
        current = path[current]
    def update(frame):
        if frame < len(path_points):
            point = path_points[frame]
            ax_larger.plot(point[1], point[0], 'ro')
            ax_larger.set_title(f'Step {frame+1}/{len(path_points)}')
        else:
            ax_larger.set_title('Large Bias Path Completed')
    anim_larger = FuncAnimation(fig_larger, update, frames=len(path_points)+1, repeat=False, interval=50)
    plt.show()

def animateForwardAS(grid, start, goal):
    fig_smaller, ax_smaller = plt.subplots()
    ax_smaller.imshow(grid, cmap='binary', interpolation='nearest')
    ax_smaller.set_title('Path Search Animation (Smaller Bias)')
    ax_smaller.set_xlabel('Columns')
    ax_smaller.set_ylabel('Rows')
    path = forwardASmall(grid, start, goal)
    path_points = []
    current = goal
    while current is not None:
        path_points.append(current)
        current = path[current]
    def update(frame):
        if frame < len(path_points):
            point = path_points[frame]
            ax_smaller.plot(point[1], point[0], 'ro')
            ax_smaller.set_title(f'Step {frame+1}/{len(path_points)}')
        else:
            ax_smaller.set_title('Small Bias Path Completed')
    anim_smaller = FuncAnimation(fig_smaller, update, frames=len(path_points)+1, repeat=False, interval=50)
    plt.show()
    
def animateBackwardA(grid, start, goal):
    fig_smaller, ax_smaller = plt.subplots()
    ax_smaller.imshow(grid, cmap='binary', interpolation='nearest')
    ax_smaller.set_title('Path Search Animation (Backward A*)')
    ax_smaller.set_xlabel('Columns')
    ax_smaller.set_ylabel('Rows')
    path = backwardA(grid, start, goal)
    path_points = []
    current = goal
    while current is not None:
        path_points.append(current)
        current = path[current]
    def update(frame):
        if frame < len(path_points):
            point = path_points[frame]
            ax_smaller.plot(point[1], point[0], 'ro')
            ax_smaller.set_title(f'Step {frame+1}/{len(path_points)}')
        else:
            ax_smaller.set_title('Backward A* Path Completed')
    anim_smaller = FuncAnimation(fig_smaller, update, frames=len(path_points)+1, repeat=False, interval=50)
    plt.show()
    
def animateAdaptiveA(grid, start, goal):
    fig_smaller, ax_smaller = plt.subplots()
    ax_smaller.imshow(grid, cmap='binary', interpolation='nearest')
    ax_smaller.set_title('Path Search Animation (Adaptive A*)')
    ax_smaller.set_xlabel('Columns')
    ax_smaller.set_ylabel('Rows')
    path = adaptiveA(grid, start, goal)
    path_points = []
    current = goal
    while current is not None:
        path_points.append(current)
        current = path[current]
    def update(frame):
        if frame < len(path_points):
            point = path_points[frame]
            ax_smaller.plot(point[1], point[0], 'ro')
            ax_smaller.set_title(f'Step {frame+1}/{len(path_points)}')
        else:
            ax_smaller.set_title('Adaptive A* Path Completed')
    anim_smaller = FuncAnimation(fig_smaller, update, frames=len(path_points)+1, repeat=False, interval=50)
    plt.show()

def main():
    gridSize = 101
    startPoint = (0, 0)
    endPoint = (gridSize - 1, gridSize - 1)
    grid = createGridworld(gridSize, startPoint, endPoint)
    print("Start Point:", startPoint)
    print("End Point:", endPoint)
    if BFS(grid, startPoint, endPoint):
        animateForwardAL(grid, startPoint, endPoint)
        animateForwardAS(grid, startPoint, endPoint)
        animateBackwardA(grid, endPoint, startPoint)
        animateAdaptiveA(grid, startPoint, endPoint)
    else:
        print("No path found.")
    
    start_time = time()
    forwardASmall(grid, startPoint, endPoint)
    end_time = time()
    print("Forward A* with smaller bias runtime:", "{:.6f}".format(end_time - start_time), "seconds")
    
    start_time = time()
    forwardALarge(grid, startPoint, endPoint)
    end_time = time()
    print("Forward A* with larger bias runtime:", "{:.6f}".format(end_time - start_time), "seconds")
    
    start_time = time()
    backwardA(grid, startPoint, endPoint)
    end_time = time()
    print("Backward A* runtime:", "{:.6f}".format(end_time - start_time), "seconds")
    
    start_time = time()
    adaptiveA(grid, startPoint, endPoint)
    end_time = time()
    print("Adaptive A* runtime:", "{:.6f}".format(end_time - start_time), "seconds")
    
if __name__ == "__main__":
    main()
