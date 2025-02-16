"""Particle filter implementation for 2D object tracking with visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

class ParticleFilter:
    def __init__(self, num_particles, space_limits, process_noise, measurement_noise):
        self.num_particles = num_particles
        self.space_limits = space_limits  # [x_min, x_max, y_min, y_max]
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Initialize particles randomly across the space
        self.particles = np.random.uniform(
            low=[space_limits[0], space_limits[2]],
            high=[space_limits[1], space_limits[3]],
            size=(num_particles, 2)
        )
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, motion):
        # Move particles according to motion model with noise
        self.particles += motion
        self.particles += np.random.normal(0, self.process_noise, self.particles.shape)
        
        # Keep particles within bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], 
                                     self.space_limits[0], 
                                     self.space_limits[1])
        self.particles[:, 1] = np.clip(self.particles[:, 1], 
                                     self.space_limits[2], 
                                     self.space_limits[3])

    def update(self, measurement):
        # Update weights based on measurement
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        self.weights = np.exp(-distances**2 / (2 * self.measurement_noise**2))
        self.weights /= np.sum(self.weights)  # Normalize weights

    def resample(self):
        # Resample particles based on weights
        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_state(self):
        # Return weighted average of particles
        return np.average(self.particles, weights=self.weights, axis=0)

def simulate_target_motion(initial_pos, steps, motion_noise):
    trajectory = np.zeros((steps, 2))
    trajectory[0] = initial_pos
    
    for t in range(1, steps):
        # Simple random walk with noise
        motion = np.random.normal(0, motion_noise, 2)
        trajectory[t] = trajectory[t-1] + motion
    
    return trajectory

# Example usage
if __name__ == "__main__":
    # Parameters
    num_particles = 1000
    space_limits = [-10, 10, -10, 10]  # [x_min, x_max, y_min, y_max]
    process_noise = 0.2
    measurement_noise = 0.5
    motion_noise = 0.1
    steps = 50

    # Initialize particle filter
    pf = ParticleFilter(num_particles, space_limits, process_noise, measurement_noise)

    # Generate true target trajectory
    true_trajectory = simulate_target_motion([0, 0], steps, motion_noise)

    # Storage for estimates
    estimated_trajectory = np.zeros((steps, 2))
    measurements = np.zeros((steps, 2))

    # Pre-compute measurements
    for t in range(steps):
        measurements[t] = true_trajectory[t] + np.random.normal(0, measurement_noise, 2)

    # Setup the figure and animation
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Calculate plot limits with padding based on true trajectory
    padding = 2  # Adjust this value to change the zoom level
    x_min, x_max = true_trajectory[:, 0].min(), true_trajectory[:, 0].max()
    y_min, y_max = true_trajectory[:, 1].min(), true_trajectory[:, 1].max()
    
    # Add padding and ensure the view window is square
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    
    ax.set_xlim(x_center - max_range/2 - padding, x_center + max_range/2 + padding)
    ax.set_ylim(y_center - max_range/2 - padding, y_center + max_range/2 + padding)
    
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.grid(True)

    # Initialize plot elements
    true_line, = ax.plot([], [], 'b-', label='True trajectory')
    estimated_line, = ax.plot([], [], 'r--', label='Estimated trajectory')
    particles_scatter = ax.scatter([], [], c='g', alpha=0.1, label='Particles')
    measurement_scatter = ax.scatter([], [], c='k', marker='x', label='Measurement')
    ax.legend()

    def init():
        true_line.set_data([], [])
        estimated_line.set_data([], [])
        particles_scatter.set_offsets(np.empty((0, 2)))
        measurement_scatter.set_offsets(np.empty((0, 2)))
        return true_line, estimated_line, particles_scatter, measurement_scatter

    def animate(frame):
        # Get measurement
        measurement = measurements[frame]
        
        # Predict step (simple random walk model)
        if frame > 0:
            motion = true_trajectory[frame] - true_trajectory[frame-1]
            pf.predict(motion)
        
        # Update step
        pf.update(measurement)
        
        # Resample
        pf.resample()
        
        # Store estimate
        estimated_trajectory[frame] = pf.estimate_state()

        # Update plot data
        true_line.set_data(true_trajectory[:frame+1, 0], true_trajectory[:frame+1, 1])
        estimated_line.set_data(estimated_trajectory[:frame+1, 0], estimated_trajectory[:frame+1, 1])
        particles_scatter.set_offsets(pf.particles)
        measurement_scatter.set_offsets(measurement.reshape(1, -1))

        ax.set_title(f'2D Particle Filter Tracking (Step {frame+1}/{steps})')
        return true_line, estimated_line, particles_scatter, measurement_scatter

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=steps,
                        interval=100, blit=True)
    
    # Save animation (optional)
    # anim.save('particle_filter.gif', writer='pillow')
    
    plt.show()
