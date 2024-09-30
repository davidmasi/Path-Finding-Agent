This project implements several pathfinding algorithms in a gridworld environment, including BFS, Forward A* with different biases, Backward A*, and Adaptive A*. The algorithms are visualized using Matplotlib animations, providing insight into how the paths are computed.

Key Features:
- Gridworld Creation: A random gridworld is generated where 0 represents walkable space, and 1 represents obstacles. Start and end points are always walkable.
- Pathfinding Algorithms:
- Breadth-First Search (BFS): Explores all possible paths uniformly.
- *Forward A (Large and Small Bias)**: A heuristic-based search where the priority can be adjusted for different biases.
- Backward A*: Similar to A* but searches from the goal towards the start.
- Adaptive A*: A dynamic version of A* that adjusts based on past exploration.
- Matplotlib Animations: The computed paths are visualized with step-by-step animations to better understand the behavior of each algorithm.
- Performance Comparison: The execution time of each algorithm is measured and displayed.

Pathfinding Algorithms Overview:
1. BFS (Breadth-First Search): A basic search algorithm that explores all possible nodes level by level from the start point.
2. *Forward A (Small and Large Bias)**:
- Uses a heuristic (Manhattan distance) to prioritize nodes closer to the goal.
- The bias modifies the weight given to the heuristic, resulting in faster or more optimal paths:
- Small Bias (1.0001): Slightly more exploration of new paths.
- Large Bias (0.9999): Aggressively favors the shortest possible route to the goal.
3. Backward A*: Starts at the goal and works backward to find the optimal path to the start point, useful in some scenarios where goal-directed search is beneficial.
4. Adaptive A*: Modifies the heuristic based on experience, allowing it to adapt dynamically to the grid as it searches for a path.

Visualizations:
The project includes animated visualizations using Matplotlib's FuncAnimation to show the step-by-step progress of the pathfinding algorithms.
- Large Bias Forward A*: The path found by Forward A* with a large bias towards optimal paths.
- Small Bias Forward A*: Visualizes the more exploratory nature of Forward A* with a smaller bias.
- Backward A*: Shows the reverse search process from the goal to the start.
- Adaptive A*: Demonstrates how the adaptive A* algorithm adjusts based on previous searches.

Commands to Run:
1. Setup and Execution:
- Ensure that numpy, matplotlib, and heapq are installed.
- Run the main() function in the script to generate the gridworld and execute the al gorithms.
- The performance of each algorithm is measured and printed in seconds.
2. Visualizing the Algorithms:
- As the algorithms run, animated visualizations are generated to depict how they traverse the gridworld and find paths.
