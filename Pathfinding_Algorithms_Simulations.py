import numpy as np
from time import time
import heapq

forwardA_smallBias = []
forwardA_largeBias = []
backward_A = []
adaptive_A = []

gridSize = 101
startPoint = (0, 0)
endPoint = (gridSize - 1, gridSize - 1)

def createGridworld(size):
    return np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])

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


def hValue(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

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
        f_value, current = heapq.heappop(tree)
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

def simulate50Times():
    storage = []
    runs = 0
    sefail = 0
    npfail = 0
    while len(storage) < 50:
        runs += 1
        grid = createGridworld(gridSize)
        if grid[startPoint] == 1 or grid[endPoint] == 1:
            sefail += 1
        else:
            if BFS(grid, startPoint, endPoint):
                start_time = time()
                parent = forwardALarge(grid, startPoint, endPoint)
                end_time = time()
                forwardA_largeBias.append(end_time - start_time)

                start_time = time()
                parent = backwardA(grid, endPoint, startPoint)
                end_time = time()
                backward_A.append(end_time - start_time)

                start_time = time()
                parent = adaptiveA(grid, startPoint, endPoint)
                end_time = time()
                adaptive_A.append(end_time - start_time)

                start_time = time()
                parent = forwardASmall(grid, startPoint, endPoint)
                end_time = time()
                forwardA_smallBias.append(end_time - start_time)

                path = []
                current = endPoint
                while current:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                tempdict = {"grid": grid, "path": path}
                storage.append(tempdict)
            else:
                npfail += 1
    print("50 worlds discovered with path after " + str(runs) + " runs\n")

simulate50Times()
print("Average runtimes:\n")
print("Forward A* (Smaller Bias): " + str(round(sum(forwardA_smallBias), 4)) + "s")
print("Forward A* (Larger Bias): " + str(round(sum(forwardA_largeBias), 4)) + "s")
print("Backward A*: " + str(round(sum(backward_A), 4)) + "s")
print("Adaptive A*: " + str(round(sum(adaptive_A), 4)) + "s")
