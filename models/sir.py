# SIR.py
# This script models and visualizes the Susceptible-Infected-Recovered (SIR) epidemic model.

import scipy.integrate as spi
import numpy as np
import pylab as pl

# --- Model Parameters Configuration ---

# Define the time range for the simulation.
t_start = 0.0  # Start time of the simulation.
ND = 60.0      # Number of Days (the total duration of the simulation).
TS = 1.0       # Time Step (the interval for calculation, e.g., 1 day).

t_end = ND
t_inc = TS
# Create an array of time points from t_start to t_end with the specified step size.
t_range = np.arange(t_start, t_end + t_inc, t_inc)

# Define the transmission and recovery rates for the SIR model.
alpha1 = 0.38  # Transmission rate (how quickly the infection spreads).
beta1 = 0.13   # Recovery rate (how quickly individuals recover).

# Define the initial state of the population.
# The format is (Susceptible, Infected, Recovered) proportions.
# The sum should be 1.0.
# Example: 80% Susceptible, 20% Infected, 0% Recovered initially.
INPUT_SIR = (0.8, 0.2, 0.0)

# --- Differential Equations for the SIR Model ---

def diff_eqs_sir(INP, t):
    """
    This function defines the system of differential equations for the SIR model.
    It calculates the rate of change for each compartment (S, I, R) at a given time t.

    Args:
        INP (list or tuple): A list containing the current proportions of S, I, and R.
                             INP[0] = S, INP[1] = I, INP[2] = R.
        t (float): The current time point (though not used in this specific system,
                   it's required by the odeint solver).

    Returns:
        np.array: An array containing the calculated rates of change [dS/dt, dI/dt, dR/dt].
    """
    # Create a zero-initialized array to store the results of the differential equations.
    Y = np.zeros((3))
    # Unpack the input state vector.
    V = INP
    
    # Equation for the rate of change of Susceptibles (dS/dt).
    # It decreases as susceptible individuals get infected.
    Y[0] = -alpha1 * V[0] * V[1]
    
    # Equation for the rate of change of Infected (dI/dt).
    # It increases with new infections and decreases as individuals recover.
    Y[1] = alpha1 * V[0] * V[1] - beta1 * V[1]
    
    # Equation for the rate of change of Recovered (dR/dt).
    # It increases as infected individuals recover.
    Y[2] = beta1 * V[1]
    
    return Y

# --- Solving the Differential Equations ---

# Use scipy's `odeint` function to solve the system of differential equations.
# It integrates the `diff_eqs_sir` function over the `t_range` with the `INPUT_SIR` initial condition.
RES2 = spi.odeint(diff_eqs_sir, INPUT_SIR, t_range)

# --- Plotting the Results ---

# Set font for the plot for a professional look.
pl.rcParams['font.family'] = 'Times New Roman'
# Create a figure with a specified size.
pl.figure(figsize=(8, 6))
pl.subplot(111)

# Plot the proportion of Susceptible individuals over time.
pl.plot(RES2[:, 0], color='y', linestyle='-', marker='*', linewidth='0.8', label='Susceptible')
# Plot the proportion of Infected individuals over time.
pl.plot(RES2[:, 1], color='#d81e06', linestyle='-', marker='*', linewidth='0.8', label='Infected')
# Plot the proportion of Recovered individuals over time.
pl.plot(RES2[:, 2], color='#008000', linestyle='-', marker='*', linewidth='0.8', label='Recovery')

# --- Annotating the Peak of the Infection ---

# Get the array of infected proportions.
infection = RES2[:, 1]
# Find the index of the maximum value in the infected array. This index corresponds to the time of the peak.
peak_time_index = np.argmax(infection)
# Get the maximum proportion of infected individuals.
peak_infection_value = infection[peak_time_index]

# Add an annotation to the plot to highlight the peak infection point.
pl.annotate(
    f'[{peak_time_index}, {peak_infection_value:.2f}]',  # The text content of the annotation, formatted to show (time, value).
    xy=(peak_time_index, peak_infection_value),         # The coordinate (x, y) that the arrow points to.
    xycoords='data',                                    # Specifies that the xy coordinates are in the data's coordinate system.
    xytext=(+40, +10),                                  # The offset of the text from the xy point.
    textcoords="offset points",                         # Specifies that the text coordinates are relative to the xy point.
    fontsize=18,                                        # Font size of the annotation text.
    color="#d81e06",                                    # Color of the annotation text and arrow.
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2") # Defines the arrow style.
)

# --- Finalizing the Plot ---

# Add a legend to the plot.
pl.legend(loc='center right', fontsize=18, bbox_to_anchor=(0.98, 0.7))
# Set the label for the x-axis.
pl.xlabel('Time', fontsize=18)
# Set the label for the y-axis.
pl.ylabel('Proportion of Population', fontsize=18) # Changed to a more descriptive label
# Set the font size for the tick labels on both axes.
pl.tick_params(labelsize=18)

# Save the plot to a PDF file with high resolution and tight bounding box.
# Make sure the target directory "../plot/picture/" exists.
# pl.savefig("../plot/picture/sir_C_m.pdf", dpi=1000, bbox_inches='tight')

# Display the plot.
pl.show()