"""
Animated comparison of Particle Filter vs Particle Gibbs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from particle_filter import ParticleFilter
from particle_gibbs import ParticleGibbs

def generate_data(steps, motion_noise, measurement_noise):
    """Generate ground truth trajectory and noisy measurements."""
    true_trajectory = np.zeros((steps, 2))
    # Add circular motion to make it more interesting
    t = np.linspace(0, 4*np.pi, steps)
    true_trajectory[:, 0] = 3 * np.cos(t)
    true_trajectory[:, 1] = 2 * np.sin(t)
    
    # Add random motion
    true_trajectory += np.cumsum(
        np.random.normal(0, motion_noise, size=true_trajectory.shape),
        axis=0
    )
    
    # Generate noisy measurements
    measurements = true_trajectory + np.random.normal(
        0, measurement_noise, size=true_trajectory.shape
    )
    
    return true_trajectory, measurements

def main():
    # Parameters
    num_particles = 100
    space_limits = [-10, 10, -10, 10]
    process_noise = 0.2
    measurement_noise = 0.5
    motion_noise = 0.1
    steps = 100
    pg_iterations = 20
    
    # Generate data
    true_trajectory, measurements = generate_data(steps, motion_noise, measurement_noise)
    
    # Initialize algorithms
    pf = ParticleFilter(num_particles, space_limits, process_noise, measurement_noise)
    pg = ParticleGibbs(num_particles, space_limits, process_noise, measurement_noise)
    
    # Run particle filter
    pf_trajectory = pf.run_filter(measurements)
    
    # Run particle Gibbs for multiple iterations
    pg_trajectories = []
    for _ in range(pg_iterations):
        pg_trajectories.append(pg.run_iteration(measurements))
    
    # Setup plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Particle Filter vs Particle Gibbs Comparison')
    
    for ax in (ax1, ax2):
        ax.set_xlim(space_limits[0], space_limits[1])
        ax.set_ylim(space_limits[2], space_limits[3])
        ax.grid(True)
        
    ax1.set_title('Particle Filter')
    ax2.set_title('Particle Gibbs')
    
    # Plot elements
    true_lines = []
    measurement_points = []
    estimate_lines = []
    
    for ax in (ax1, ax2):
        true_lines.append(ax.plot([], [], 'b-', label='True', alpha=0.5)[0])
        measurement_points.append(ax.plot([], [], 'kx', label='Measurements')[0])
        estimate_lines.append(ax.plot([], [], 'r-', label='Estimate')[0])
        ax.legend()
    
    def init():
        for line in true_lines + measurement_points + estimate_lines:
            line.set_data([], [])
        return true_lines + measurement_points + estimate_lines
    
    def animate(frame):
        # Update data up to current frame
        t = frame + 1
        
        for i, ax in enumerate((ax1, ax2)):
            # Plot true trajectory and measurements
            true_lines[i].set_data(true_trajectory[:t, 0], true_trajectory[:t, 1])
            measurement_points[i].set_data(measurements[:t, 0], measurements[:t, 1])
            
            # Plot estimates
            if i == 0:  # Particle Filter
                estimate_lines[i].set_data(pf_trajectory[:t, 0], pf_trajectory[:t, 1])
            else:  # Particle Gibbs (show latest iteration)
                estimate_lines[i].set_data(
                    pg_trajectories[-1][:t, 0],
                    pg_trajectories[-1][:t, 1]
                )
        
        return true_lines + measurement_points + estimate_lines
    
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=steps,
        interval=50, blit=True
    )
    
    plt.show()

if __name__ == "__main__":
    main() 