"""
Side-by-side comparison of Particle Filtering and Particle Gibbs for trajectory estimation.
Shows how MCMC-based particle methods (PG) compare to sequential particle filtering (PF).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from metropolis_hastings import simulate_data

class ParticleFilter:
    def __init__(self, n_particles, process_noise):
        self.n_particles = n_particles
        self.process_noise = process_noise
        self.particles = None
        self.weights = None
        
    def initialize(self, initial_state):
        """Initialize particles around initial state."""
        self.particles = initial_state + np.random.normal(0, self.process_noise, 
                                                        (self.n_particles, 2))
        self.weights = np.ones(self.n_particles) / self.n_particles
        
    def update(self, observation, motion=None):
        """Update particles using motion model and observation."""
        # Apply motion if provided
        if motion is not None:
            self.particles = self.particles + motion
        
        # Add process noise
        self.particles += np.random.normal(0, self.process_noise, self.particles.shape)
        
        # Update weights based on observation likelihood
        distances = np.linalg.norm(self.particles - observation, axis=1)
        self.weights = np.exp(-distances**2 / (2 * self.process_noise**2))
        self.weights /= np.sum(self.weights)
        
        # Resample if effective sample size is too low
        n_eff = 1 / np.sum(self.weights**2)
        if n_eff < self.n_particles / 2:
            indices = np.random.choice(self.n_particles, self.n_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles
            
    def get_estimate(self):
        """Get current state estimate."""
        return np.average(self.particles, weights=self.weights, axis=0)

class ParticleGibbs:
    def __init__(self, n_particles, process_noise):
        self.n_particles = n_particles
        self.process_noise = process_noise
        
    def run_iteration(self, observations, conditioned_trajectory=None):
        """
        Run one iteration of Particle Gibbs with ancestor sampling.
        This implementation forces the conditioned state into the particle set at each time step,
        and stores the partial (per-time-step) particle weights so that backward sampling uses
        only the weights computed in the current forward pass.
        """
        steps = len(observations)
        pf = ParticleFilter(self.n_particles, self.process_noise)
        
        # Initialize particles at t = 0
        pf.initialize(observations[0])
        if conditioned_trajectory is not None:
            # Force the conditioned trajectory at time 0
            pf.particles[0] = conditioned_trajectory[0]
        all_particles = [pf.particles.copy()]
        all_weights = [pf.weights.copy()]
        
        # Forward pass: build partial trajectories and save weights at each time step
        for t in range(1, steps):
            # Compute motion from the observations and update particles with noise.
            motion = observations[t] - observations[t-1]
            pf.particles = pf.particles + motion
            pf.particles += np.random.normal(0, self.process_noise, pf.particles.shape)
            
            # IMPORTANT: Force the conditioned trajectory for the current time step
            if conditioned_trajectory is not None:
                pf.particles[0] = conditioned_trajectory[t]
                
            # Compute observation likelihood for current particles
            distances = np.linalg.norm(pf.particles - observations[t], axis=1)
            w = np.exp(- distances**2 / (2 * self.process_noise**2))
            w /= np.sum(w)
            pf.weights = w
            
            # Conditional resampling with ancestor sampling if effective sample size is low.
            n_eff = 1 / np.sum(pf.weights**2)
            if n_eff < self.n_particles / 2:
                if conditioned_trajectory is not None:
                    # Save the conditioned particle from the previous (expected) value.
                    saved_particle = conditioned_trajectory[t]
                    # (Optionally, save the local weight of index 0 in the current particle set)
                    # But then compute ancestor sampling weights using current partial weights:
                    ancestor_weights = pf.weights.copy()
                    # Compute transition likelihood: how likely is it to transition from each particle to the conditioned state:
                    next_dists = np.sum((pf.particles - saved_particle)**2, axis=1)
                    transition_logweights = - next_dists / (2 * self.process_noise**2)
                    
                    # Incorporate a "future" likelihood if available.
                    if t < steps - 1:
                        # Here we approximate future likelihood via the conditioned trajectory differences.
                        future_traj = conditioned_trajectory[t:]
                        # (For simplicity, we sum the squared differences for the future trajectory.)
                        future_dists = np.sum((future_traj[1:] - future_traj[:-1])**2, axis=1)
                        future_logweights = - np.sum(future_dists) / (2 * self.process_noise**2)
                        transition_logweights += future_logweights
                        
                    # Combine with current particle weights.
                    ancestor_weights *= np.exp(transition_logweights - np.max(transition_logweights))
                    ancestor_weights /= np.sum(ancestor_weights)
                    
                    # Sample an ancestor index based on the computed weights.
                    ancestor_idx = np.random.choice(self.n_particles, p=ancestor_weights)
                    ancestor_weight = pf.weights[ancestor_idx]
                    
                # Resample using the current weights
                indices = np.random.choice(self.n_particles, self.n_particles, p=pf.weights)
                pf.particles = pf.particles[indices]
                pf.weights = np.ones(self.n_particles) / self.n_particles
                
                # Restore the conditioned trajectory with the sampled (ancestor's) weight.
                if conditioned_trajectory is not None:
                    pf.particles[0] = saved_particle
                    pf.weights[0] = ancestor_weight
                    pf.weights = pf.weights / np.sum(pf.weights)
            
            # Record the partial particle set and weights for this time step.
            all_particles.append(pf.particles.copy())
            all_weights.append(pf.weights.copy())
        
        # Backward sampling: use the stored per-time-step (partial) weights.
        trajectory = np.zeros((steps, 2))
        final_w = all_weights[-1] / np.sum(all_weights[-1])
        idx = np.random.choice(self.n_particles, p=final_w)
        trajectory[-1] = all_particles[-1][idx]
        for t in range(steps - 2, -1, -1):
            deltas = trajectory[t+1] - all_particles[t]
            log_w = - np.sum(deltas**2, axis=1) / (2 * self.process_noise**2)
            log_w -= np.max(log_w)  # Numerical stability
            w = np.exp(log_w) * all_weights[t]
            w /= np.sum(w)
            idx = np.random.choice(self.n_particles, p=w)
            trajectory[t] = all_particles[t][idx]
        
        return trajectory

def main():
    # Parameters
    steps = 50
    process_noise = 0.1
    motion_noise = 0.05
    measurement_noise = 0.05  # Reduced from 0.2 to 0.05 for clearer measurements
    n_particles = 200
    pg_iterations = 100
    
    # Generate true trajectory and measurements
    true_trajectory = simulate_data(steps, motion_noise)
    measurements = true_trajectory + np.random.normal(0, measurement_noise, true_trajectory.shape)  # Added
    
    # Run particle filter
    pf = ParticleFilter(n_particles, process_noise)
    pf.initialize(true_trajectory[0])
    pf_estimates = [pf.get_estimate()]
    pf_particles = [pf.particles.copy()]
    
    for t in range(1, steps):
        motion = true_trajectory[t] - true_trajectory[t-1]
        pf.update(measurements[t], motion)  # Changed: use measurements
        pf_estimates.append(pf.get_estimate())
        pf_particles.append(pf.particles.copy())
    
    # Run particle Gibbs
    pg = ParticleGibbs(n_particles, process_noise)
    pg_trajectories = []
    current_trajectory = measurements  # Changed: initialize with measurements
    
    for i in range(pg_iterations):
        current_trajectory = pg.run_iteration(measurements, current_trajectory)  # Changed: use measurements
        pg_trajectories.append(current_trajectory)
    
    # Setup visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Particle Methods Comparison: Sequential vs MCMC', 
                fontsize=16, y=0.95)
    
    # Setup plot limits
    padding = 0.5
    x_min, x_max = true_trajectory[:, 0].min(), true_trajectory[:, 0].max()
    y_min, y_max = true_trajectory[:, 1].min(), true_trajectory[:, 1].max()
    max_range = max(x_max - x_min, y_max - y_min)
    center = np.array([(x_max + x_min)/2, (y_max + y_min)/2])
    
    margin = max_range * 0.1
    max_range += margin
    
    for ax in [ax1, ax2]:
        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.grid(True)
        ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-', 
               label='True trajectory', alpha=0.5)
    
    # Particle Filter plot elements
    particles_scatter = ax1.scatter([], [], c='r', alpha=0.2, s=10)
    pf_line, = ax1.plot([], [], 'g-', label='PF estimate', linewidth=2)
    
    # Particle Gibbs plot elements
    pg_lines = []
    for _ in range(pg_iterations):
        line, = ax2.plot([], [], 'r-', alpha=0.1)
        pg_lines.append(line)
    pg_current_line, = ax2.plot([], [], 'g-', label='Current', linewidth=2)
    
    # Add info text
    pf_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    pg_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set titles
    ax1.set_title('Sequential Particle Filter\nOnline State Estimation', pad=20)
    ax2.set_title('Particle Gibbs\nMCMC Trajectory Sampling', pad=20)
    
    # Remove the final PG trajectory from both plots
    for ax in [ax1, ax2]:
        ax.legend(loc='lower right')
    
    # Add measurement scatter plots to both axes
    measurement_scatter1 = ax1.scatter([], [], c='k', marker='x', s=30, 
                                     label='Measurements', alpha=0.5)
    measurement_scatter2 = ax2.scatter([], [], c='k', marker='x', s=30, 
                                     label='Measurements', alpha=0.5)
    
    def init():
        particles_scatter.set_offsets(np.empty((0, 2)))
        pf_line.set_data([], [])
        measurement_scatter1.set_offsets(np.empty((0, 2)))  # Added
        measurement_scatter2.set_offsets(np.empty((0, 2)))  # Added
        for line in pg_lines:
            line.set_data([], [])
        pg_current_line.set_data([], [])
        pf_text.set_text('')
        pg_text.set_text('')
        return ([particles_scatter, pf_line, measurement_scatter1, measurement_scatter2] + 
                pg_lines + [pg_current_line, pf_text, pg_text])
    
    def animate(frame):
        # Compute current step and iteration for PG
        current_iter = min(frame // steps, pg_iterations-1)
        time_steps = min(frame % steps + 1, steps)
        
        # Update Particle Filter visualization
        if frame < steps:
            if frame > 0:
                particles_scatter.set_offsets(pf_particles[frame])
                pf_line.set_data([p[0] for p in pf_estimates[:frame+1]],
                                [p[1] for p in pf_estimates[:frame+1]])
                
                # Update measurements
                measurement_scatter1.set_offsets(measurements[:frame+1])
                measurement_scatter2.set_offsets(measurements[:frame+1])
                
                pf_error = np.mean([np.sum((est - true)**2) 
                                  for est, true in zip(pf_estimates[:frame+1],
                                                     true_trajectory[:frame+1])])
                pf_text.set_text(f'Step: {frame + 1}/{steps}\n'
                               f'MSE: {pf_error:.3f}\n'
                               f'Particles: {n_particles}\n'
                               f'Sequential Update')
        
        # Update Particle Gibbs visualization
        if frame < total_frames - steps:  # During iterations
            for i in range(current_iter):
                pg_lines[i].set_data(pg_trajectories[i][:time_steps, 0],
                                   pg_trajectories[i][:time_steps, 1])
                pg_lines[i].set_alpha(0.1)
            
            # Show current iteration
            if current_iter < pg_iterations:
                pg_current_line.set_data(pg_trajectories[current_iter][:time_steps, 0],
                                       pg_trajectories[current_iter][:time_steps, 1])
                pg_current_line.set_color('r')  # Use red during iterations
                
                pg_error = np.mean((pg_trajectories[current_iter][:time_steps] - 
                                  true_trajectory[:time_steps])**2)
                pg_text.set_text(f'Iteration: {current_iter + 1}/{pg_iterations}\n'
                               f'Step: {time_steps}/{steps}\n'
                               f'Current MSE: {pg_error:.3f}\n'
                               f'Particles: {n_particles}')
        else:  # After all iterations
            # Clear all previous trajectories
            for line in pg_lines:
                line.set_data([], [])
            
            # Show only final trajectory in green
            pg_current_line.set_data(pg_trajectories[-1][:, 0], pg_trajectories[-1][:, 1])
            pg_current_line.set_color('g')  # Change to green for final result
            
            final_error = np.mean((pg_trajectories[-1] - true_trajectory)**2)
            pg_text.set_text(f'Final Result\n'
                           f'MSE: {final_error:.3f}\n'
                           f'Particles: {n_particles}\n'
                           f'Iterations: {pg_iterations}')
        
        return ([particles_scatter, pf_line, measurement_scatter1, measurement_scatter2] + 
                pg_lines + [pg_current_line, pf_text, pg_text])
    
    # Create animation that shows full PG iterations
    total_frames = steps * pg_iterations
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=total_frames,
                        interval=50, blit=True, repeat=False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 