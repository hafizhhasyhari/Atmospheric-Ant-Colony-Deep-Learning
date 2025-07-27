import numpy as np
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

# Generate random city coordinates
num_cities = 10
cities = np.random.rand(num_cities, 2) * 100

# Parameter ACO
num_ants = 20
num_iterations = 100
alpha = 1.0        # pheromone importance
beta = 5.0         # distance importance
evaporation = 0.5
pheromone_deposit = 100.0

# Initialize pheromone matrix and distance matrix
pheromones = np.ones((num_cities, num_cities))
distances = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        distances[i][j] = np.linalg.norm(cities[i] - cities[j])

def select_next_city(current_city, visited, pheromones, distances):
    probabilities = []
    for city in range(num_cities):
        if city not in visited:
            tau = pheromones[current_city][city] ** alpha
            eta = (1 / distances[current_city][city]) ** beta
            probabilities.append(tau * eta)
        else:
            probabilities.append(0)
    probabilities = np.array(probabilities)
    if probabilities.sum() == 0:
        return np.random.choice([c for c in range(num_cities) if c not in visited])
    probabilities /= probabilities.sum()
    return np.random.choice(range(num_cities), p=probabilities)

def run_aco():
    global pheromones, best_path, best_distance, convergence
    best_distance = float('inf')
    best_path = None
    convergence = []

    for iteration in range(num_iterations):
        all_paths = []
        all_distances = []

        for _ in range(num_ants):
            path = [np.random.randint(num_cities)]
            while len(path) < num_cities:
                next_city = select_next_city(path[-1], path, pheromones, distances)
                path.append(next_city)
            all_paths.append(path)

            # Calculate path distance
            dist = sum(distances[path[i]][path[(i + 1) % num_cities]] for i in range(num_cities))
            all_distances.append(dist)

            # Update best path
            if dist < best_distance:
                best_distance = dist
                best_path = path

        # Evaporate pheromones
        pheromones *= (1 - evaporation)

        # Deposit new pheromones
        for path, dist in zip(all_paths, all_distances):
            for i in range(num_cities):
                from_city = path[i]
                to_city = path[(i + 1) % num_cities]
                pheromones[from_city][to_city] += pheromone_deposit / dist

        convergence.append(best_distance)

# --- Visualization Functions ---

def plot_initial_cities():
    plt.figure(figsize=(6, 5))
    plt.scatter(cities[:, 0], cities[:, 1], c='blue', label='City')
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, f"{i}", fontsize=9)
    plt.title("Initial City Layout")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_path(path, title="Path"):
    plt.figure(figsize=(6, 5))
    for i in range(len(path)):
        city_from = cities[path[i]]
        city_to = cities[path[(i+1) % len(path)]]
        plt.plot([city_from[0], city_to[0]], [city_from[1], city_to[1]], 'r-')
    plt.scatter(cities[:, 0], cities[:, 1], c='blue')
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, f"{i}", fontsize=9)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def plot_convergence(convergence):
    plt.figure(figsize=(7, 5))
    plt.plot(convergence, label='Best Distance', color='green')
    plt.title("Convergence of ACO")
    plt.xlabel("Iteration")
    plt.ylabel("Best Distance Found")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pheromone_map():
    plt.figure(figsize=(6, 5))
    plt.imshow(pheromones, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Pheromone Intensity')
    plt.title("Final Pheromone Map")
    plt.xlabel("City")
    plt.ylabel("City")
    plt.show()

# --- Main Execution ---
plot_initial_cities()                 # Visualization 1
run_aco()
plot_path(best_path, "Best Path")     # Visualization 2
plot_convergence(convergence)         # Visualization 3
plot_pheromone_map()                  # Visualization 4
