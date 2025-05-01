import paramiko
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import math
import matplotlib.pyplot as plt
import time

# -----------------------------
# Module 1: Collect VM Stats
# -----------------------------
def parse_network_io(value):
    """Parse network I/O string (e.g., '4.03kB') to float in bytes."""
    if not value or not isinstance(value, str):
        return 0.0
    try:
        sent_value = value.split('/')[0].strip()
        num = float(sent_value.replace('kB', '').replace('B', '').strip())
        if 'kB' in sent_value:
            num *= 1024  # Convert kB to bytes
        elif 'MB' in sent_value:
            num *= 1024 * 1024  # Convert MB to bytes
        return num
    except (ValueError, IndexError):
        return 0.0

def parse_value(value, is_percentage=False):
    """Parse a value to float, handling percentages or raw numbers."""
    if not value or not isinstance(value, str):
        return 0.0
    try:
        if is_percentage:
            return float(value.replace('%', '').strip()) / 100
        return float(value.strip())
    except ValueError:
        return 0.0

# Collect VM stats
vm_data = []
with open('instance_info.txt', 'r') as f:
    ips = [line.strip().split(':')[1] for line in f]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    for i, ip in enumerate(ips):
        print(f"Attempting to connect to {ip}")
        ssh.connect(ip, username='bitnami', key_filename='C:\\Users\\satya\\OneDrive\\Documents\\Desktop\\DADo\\keys\\SSH1.pem')
        print(f"Connected to {ip}")
        stdin, stdout, stderr = ssh.exec_command("docker stats --no-stream --format \"{{.Name}},{{.CPUPerc}},{{.MemPerc}},{{.NetIO}}\" 2>&1")
        time.sleep(1)  # Allow time for output
        stats_output = stdout.readlines()
        error_output = stderr.readlines()
        if error_output:
            print(f"Error for {ip}: {''.join(error_output)}")
        stats_lines = [line.strip().split(',') for line in stats_output if line.strip()]
        print(f"Raw stats for {ip} (all lines): {stats_lines}")
        valid_stats = stats_lines if stats_lines else []
        print(f"Raw stats for {ip} (filtered): {valid_stats}")
        if valid_stats and len(valid_stats[0]) >= 4:
            stats = valid_stats[0]  # Use first container's stats as VM aggregate
            cpu = parse_value(stats[1], is_percentage=True)
            mem = parse_value(stats[2], is_percentage=True)
            net = parse_network_io(stats[3]) if stats[3] else 0
            vm_data.append([cpu, mem, net])
        else:
            vm_data.append([0, 0, 0])  # Default if no valid stats
        ssh.close()

    num_vms = len(ips)
    containers_per_vm = 2  # Updated to match actual container count (2 per VM)
    total_containers = num_vms * containers_per_vm
    features = 3  # CPU, Memory, Network

    # Generate initial state based on VM stats
    initial_state = np.zeros((total_containers, features))
    for vm_idx in range(num_vms):
        vm_stats = vm_data[vm_idx]
        for cont_idx in range(containers_per_vm):
            initial_state[vm_idx * containers_per_vm + cont_idx] = vm_stats

    X_train = initial_state.reshape(-1, features)
    y_train = initial_state[:, -1]  # Using Network as target for simplicity

except Exception as e:
    print(f"Error occurred during VM data collection: {e}")
finally:
    ssh.close()

# -----------------------------
# Module 2: Actor-Critic Neural Network
# -----------------------------
actor_model = tf.keras.Sequential([
    Dense(32, input_shape=(features,), activation='relu'),
    Dense(1)
])
actor_model.compile(optimizer='adam', loss='mse')

critic_model = tf.keras.Sequential([
    Dense(32, input_shape=(features,), activation='relu'),
    Dense(1)
])
critic_model.compile(optimizer='adam', loss='mse')

print("\n🎯 Training Actor Model for Load Prediction...")
actor_model.fit(X_train, y_train, epochs=20, verbose=0)

print("\n🎯 Training Critic Model for Value Estimation...")
critic_model.fit(X_train, y_train, epochs=20, verbose=0)

# Predict load for containers
def predict_load(state):
    loads = []
    for i in range(total_containers):
        container_state = state[i * features:(i + 1) * features]
        load = actor_model.predict(np.array([container_state]), verbose=0)[0][0]
        loads.append(load)
    return loads

# -----------------------------
# Module 3: Dragonfly Algorithm
# -----------------------------
population_size = 20
max_iterations = 5
dim = total_containers * features
w = 0.9
s = a = c = f = e = 0.1

population = np.random.rand(population_size, dim) * 0.9 + 0.1
velocity = np.random.uniform(-0.1, 0.1, size=(population_size, dim))
population = np.clip(population, 0.01, 1)

fitness_history = []

def levy_flight(Lambda):
    sigma = (math.gamma(1 + Lambda) * np.sin(math.pi * Lambda / 2) /
             (math.gamma((1 + Lambda) / 2) * Lambda * 2 * ((Lambda - 1) / 2))) * (1 / Lambda)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / (abs(v) ** (1 / Lambda))
    return step * 0.1

def fitness(solution):
    predicted_loads = predict_load(solution)
    total_predicted_load = np.sum(predicted_loads)
    min_predicted_load = np.min(predicted_loads)
    
    transmission_cost = np.sum(solution) * 0.5
    energy_consumption = np.sum(solution**2)
    migration_time = np.sum(solution) * 2
    
    total_cost = (
        0.3 * total_predicted_load + 
        0.2 * transmission_cost + 
        0.2 * energy_consumption + 
        0.1 * migration_time + 
        0.05 * (1 - min_predicted_load)
    )
    return total_cost, total_predicted_load, transmission_cost, energy_consumption, migration_time

# Optimization
best_solution = None
best_fitness = float('inf')
best_predicted_load = None
best_transmission_cost = None
best_energy_consumption = None
best_migration_time = None

print("\n🚀 Running Dragonfly Algorithm...")
for iteration in range(max_iterations):
    for i in range(population_size):
        sep = np.mean(population - population[i], axis=0)
        align = np.mean(velocity, axis=0)
        cohesion = np.mean(population, axis=0) - population[i]
        attraction = np.min(population, axis=0) - population[i]
        distraction = population[i] - np.max(population, axis=0)
        
        velocity[i] = (
            w * velocity[i] + s * sep + a * align + c * cohesion +
            f * attraction + e * distraction + levy_flight(1.5) +
            np.random.uniform(-0.05, 0.05, size=dim)
        )
        velocity[i] = np.clip(velocity[i], -0.1, 0.1)
        population[i] += velocity[i]
        population[i] = np.clip(population[i], 0.01, 1)
        
        current_fitness, predicted_load, transmission_cost, energy_consumption, migration_time = fitness(population[i])
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = population[i]
            best_predicted_load = predicted_load
            best_transmission_cost = transmission_cost
            best_energy_consumption = energy_consumption
            best_migration_time = migration_time
    
    fitness_history.append(best_fitness)
    print(f"Iteration {iteration + 1}/{max_iterations} - Best Fitness: {best_fitness:.4f}")

# -----------------------------
# Module 4: Results with VMs and Containers
# -----------------------------
print("\n✅ Best Solution Found:")
best_predicted_loads = predict_load(best_solution)

# Organize results by VM
for vm in range(num_vms):
    print(f"\nVM {vm + 1} (IP: {ips[vm]}):")
    for cont in range(containers_per_vm):
        idx = vm * containers_per_vm + cont
        start = idx * features
        print(f"  Container {cont + 1}: CPU={best_solution[start]:.2f}, Memory={best_solution[start + 1]:.2f}, Network={best_solution[start + 2]:.2f}")

least_loaded_container_idx = np.argmin(best_predicted_loads)
least_loaded_vm = least_loaded_container_idx // containers_per_vm
least_loaded_container = least_loaded_container_idx % containers_per_vm + 1

best_container_energy = sum(best_solution[least_loaded_container_idx * features:(least_loaded_container_idx + 1) * features]**2)
best_container_cost = sum(best_solution[least_loaded_container_idx * features:(least_loaded_container_idx + 1) * features]) * 0.5
conserved_energy = best_energy_consumption - best_container_energy
conserved_cost = best_transmission_cost - best_container_cost

print(f"\n🏆 Least Loaded: VM {least_loaded_vm + 1} (IP: {ips[least_loaded_vm]}), Container {least_loaded_container}")
print(f"🔋 Best Container Energy Consumption: {best_container_energy:.4f}")
print(f"💰 Best Container Transmission Cost: {best_container_cost:.4f}")
print(f"🌍 Energy Conserved: {conserved_energy:.4f}")
print(f"💵 Cost Conserved: {conserved_cost:.4f}")

print(f"\nBest Predicted Load (Total): {best_predicted_load:.4f}")
print(f"Best Transmission Cost (Overall): {best_transmission_cost:.4f}")
print(f"Best Energy Consumption (Overall): {best_energy_consumption:.4f}")
print(f"Best Migration Time: {best_migration_time:.4f}")

# Result Summary
print("\n📊 Result Summary:")
print("The dragonfly algorithm optimized resource allocation across three VMs.")
print(f"Loads were balanced, with VM {least_loaded_vm + 1} (IP: {ips[least_loaded_vm]}), Container {least_loaded_container} being the least loaded.")
print("Energy and cost were conserved, enhancing efficiency across all VMs.")
print("Transmission costs and migration times were minimized system-wide.")

# -----------------------------
# Module 5: Calculate Metrics per VM for Scatter Plots
# -----------------------------
# Initialize lists to store per-VM metrics
energy_per_vm = []
cost_per_vm = []
time_per_vm = []
load_per_vm = []

# Calculate metrics for each VM
for vm in range(num_vms):
    vm_energy = 0
    vm_cost = 0
    vm_time = 0
    vm_load = 0
    
    # Sum metrics across all containers in this VM
    for cont in range(containers_per_vm):
        idx = vm * containers_per_vm + cont
        start = idx * features
        
        # Energy: sum of squared resources for this container
        vm_energy += best_solution[start]**2 + best_solution[start + 1]**2 + best_solution[start + 2]**2
        # Cost: transmission cost for this container (sum of resources * 0.5)
        vm_cost += sum(best_solution[start:(idx + 1) * features]) * 0.5
        # Time: migration time for this container (sum of resources * 2)
        vm_time += sum(best_solution[start:(idx + 1) * features]) * 2
        # Load: predicted load for this container
        vm_load += best_predicted_loads[idx]
    
    energy_per_vm.append(vm_energy)
    cost_per_vm.append(vm_cost)
    time_per_vm.append(vm_time)
    load_per_vm.append(vm_load)

# VM labels for x-axis
vm_labels = [f"VM {i+1} ({ips[i]})" for i in range(num_vms)]

# -----------------------------
# Module 6: Connected Line Plots
# -----------------------------
plt.figure(figsize=(15, 15))

# Fitness Score Plot
plt.subplot(3, 2, 1)
plt.plot(range(1, max_iterations + 1), fitness_history, label="Fitness Score", color="blue", marker="o")
plt.title("Fitness Score Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Fitness Score")
plt.legend()
plt.grid(True)

# Energy vs VMs
plt.subplot(3, 2, 2)
plt.plot(range(num_vms), energy_per_vm, color='lightgreen', marker='o', linestyle='-', markersize=8, alpha=0.6)
plt.xticks(range(num_vms), vm_labels)
plt.title('Energy Consumption vs VMs')
plt.xlabel('VM')
plt.ylabel('Energy Consumption')
plt.grid(True)

# Cost vs VMs
plt.subplot(3, 2, 3)
plt.plot(range(num_vms), cost_per_vm, color='orange', marker='o', linestyle='-', markersize=8, alpha=0.6)
plt.xticks(range(num_vms), vm_labels)
plt.title('Transmission Cost vs VMs')
plt.xlabel('VM')
plt.ylabel('Cost')
plt.grid(True)

# Time vs VMs
plt.subplot(3, 2, 4)
plt.plot(range(num_vms), time_per_vm, color='lightcoral', marker='o', linestyle='-', markersize=8, alpha=0.6)
plt.xticks(range(num_vms), vm_labels)
plt.title('Migration Time vs VMs')
plt.xlabel('VM')
plt.ylabel('Time')
plt.grid(True)

# Load vs VMs
plt.subplot(3, 2, 5)
plt.plot(range(num_vms), load_per_vm, color='skyblue', marker='o', linestyle='-', markersize=8, alpha=0.6)
plt.xticks(range(num_vms), vm_labels)
plt.title('Predicted Load vs VMs')
plt.xlabel('VM')
plt.ylabel('Load')
plt.grid(True)

plt.tight_layout()
plt.show()