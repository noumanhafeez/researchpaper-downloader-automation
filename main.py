import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Simulation Parameters
# -----------------------------
high_priority = 5
regular_priority = 25
low_priority = 50

total_devices = high_priority + regular_priority + low_priority  # 80

Nm = 8     # mini-slots per time slot
Ns = 10    # time slots per frame
N = 10000  # number of frames

# -----------------------------
# Function to simulate one frame
# -----------------------------
def simulate_frame(P):
    # Step 1: Decide active devices
    active_devices = np.random.rand(total_devices) < P
    active_count = np.sum(active_devices)

    if active_count == 0:
        return 0

    successful_transmissions = 0

    # Step 2: For each time slot
    for _ in range(Ns):
        # Step 3: Each active device selects a mini-slot
        mini_slot_choices = np.random.randint(0, Nm, active_count)

        # Step 4: Count collisions
        unique, counts = np.unique(mini_slot_choices, return_counts=True)

        # Step 5: Success if only one device in mini-slot
        successes = np.sum(counts == 1)
        successful_transmissions += successes

    return successful_transmissions


# -----------------------------
# Main Simulation
# -----------------------------
P_values = np.linspace(0.05, 1.0, 10)  # different probabilities
Sd_values = []

for P in P_values:
    total_success = 0

    for _ in range(N):
        Yi = simulate_frame(P)
        total_success += Yi

    Sd = total_success / N
    Sd_values.append(Sd)
    print(f"P = {P:.2f}, Sd = {Sd:.2f}")


# -----------------------------
# Plot Sd vs P
# -----------------------------
plt.figure()
plt.plot(P_values, Sd_values, marker='o')
plt.xlabel("Probability (P)")
plt.ylabel("Avg Successful Devices per Frame (Sd)")
plt.title("Sd vs P for IoT MAC Protocol Simulation")
plt.grid()

# ✅ Save the plot
plt.savefig("Sd_vs_P.png", dpi=300)

plt.show()



'''

Output of code in terminal

P = 0.05, Sd = 24.32
P = 0.16, Sd = 26.35
P = 0.26, Sd = 15.17
P = 0.37, Sd = 7.17
P = 0.47, Sd = 3.13
P = 0.58, Sd = 1.24
P = 0.68, Sd = 0.47
P = 0.79, Sd = 0.17
P = 0.89, Sd = 0.06
P = 1.00, Sd = 0.02

'''