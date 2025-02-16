"""
Parallel Tempering (Replica Exchange MCMC) for 2D trajectory estimation.

Key components:
1. Multiple chains at different temperatures
2. Higher temperatures allow for easier barrier crossing
3. Exchange between chains helps escape local optima
4. Final samples taken from lowest temperature chain
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def simulate_data(steps, motion_noise):
    """Simulate complex ground truth trajectory with non-looping paths."""
    # Generate base trajectory with varying velocities
    t = np.linspace(0, 1, steps)
    
    # Random coefficients for polynomial components
    coeffs_x = np.random.normal(0, 1, 4)
    coeffs_y = np.random.normal(0, 1, 4)
    
    # Create smooth trajectory using polynomial basis
    x = (coeffs_x[0] + 
         coeffs_x[1] * t + 
         coeffs_x[2] * t**2 + 
         coeffs_x[3] * t**3)
    y = (coeffs_y[0] + 
         coeffs_y[1] * t + 
         coeffs_y[2] * t**2 + 
         coeffs_y[3] * t**3)
    
    # Create trajectory
    true_trajectory = np.column_stack([x, y])
    
    # Add smooth noise
    noise = np.random.normal(0, motion_noise, (steps, 2))
    window = 5  # Smoothing window
    smoothed_noise = np.array([np.convolve(noise[:, i], 
                                          np.ones(window)/window, 
                                          mode='same') 
                              for i in range(2)]).T
    
    true_trajectory += smoothed_noise
    
    # Center and scale
    true_trajectory -= np.mean(true_trajectory, axis=0)
    scale = np.random.uniform(1.5, 2.5)
    true_trajectory *= scale
    
    return true_trajectory

class ParallelTempering:
    def __init__(self, true_trajectory, process_noise, n_chains=4, max_temp=10.0):
        self.true_trajectory = true_trajectory
        self.process_noise = process_noise
        self.steps = len(true_trajectory)
        self.n_chains = n_chains
        
        # Set up temperature ladder (geometric progression)
        self.temperatures = np.exp(np.linspace(0, np.log(max_temp), n_chains))
        
    def log_likelihood(self, trajectory, temperature=1.0):
        """Compute tempered log likelihood."""
        # Direct comparison with true trajectory
        trajectory_error = np.sum((trajectory - self.true_trajectory)**2)
        log_target = -trajectory_error / (2 * self.process_noise**2 * temperature)
        
        # Process likelihood (smoothness prior)
        velocities = trajectory[1:] - trajectory[:-1]
        process_error = np.sum(velocities**2)
        log_process = -process_error / (2 * self.process_noise**2 * temperature)
            
        return log_target + log_process
    
    def propose_trajectory(self, current_trajectory, temperature, proposal_std=0.1):
        """Propose new trajectory with temperature-scaled noise."""
        # Scale proposal by temperature for better mixing at higher temperatures
        scaled_std = proposal_std * np.sqrt(temperature)
        noise = np.random.normal(0, scaled_std, current_trajectory.shape)
        noise = np.cumsum(noise, axis=0) * 0.1
        
        return current_trajectory + noise
    
    def run_chain(self, initial_trajectories, max_iterations=1000, proposal_std=0.1,
                 error_threshold=0.1, patience=50, swap_interval=10):
        """Run parallel tempering chain."""
        current_trajectories = [traj.copy() for traj in initial_trajectories]
        current_log_probs = [self.log_likelihood(traj, temp) 
                           for traj, temp in zip(current_trajectories, self.temperatures)]
        
        # Storage for chain 0 (lowest temperature)
        trajectories = [current_trajectories[0].copy()]
        accepted = []
        errors = []
        swaps = []  # Track successful swaps
        
        best_error = float('inf')
        iterations_without_improvement = 0
        
        for i in range(max_iterations):
            # Regular MCMC updates for each chain
            for j in range(self.n_chains):
                proposed = self.propose_trajectory(
                    current_trajectories[j], 
                    self.temperatures[j], 
                    proposal_std
                )
                proposed_log_prob = self.log_likelihood(proposed, self.temperatures[j])
                
                # Accept/reject
                log_ratio = proposed_log_prob - current_log_probs[j]
                if np.log(np.random.random()) < log_ratio:
                    current_trajectories[j] = proposed
                    current_log_probs[j] = proposed_log_prob
                    if j == 0:  # Track acceptance for base chain only
                        accepted.append(True)
                else:
                    if j == 0:
                        accepted.append(False)
            
            # Attempt temperature swaps
            if i % swap_interval == 0:
                for j in range(self.n_chains - 1):
                    # Compute swap acceptance probability
                    log_prob_i = self.log_likelihood(current_trajectories[j], 
                                                   self.temperatures[j+1])
                    log_prob_j = self.log_likelihood(current_trajectories[j+1], 
                                                   self.temperatures[j])
                    log_ratio = (log_prob_i + log_prob_j - 
                               current_log_probs[j] - current_log_probs[j+1])
                    
                    if np.log(np.random.random()) < log_ratio:
                        # Swap trajectories
                        current_trajectories[j], current_trajectories[j+1] = \
                            current_trajectories[j+1], current_trajectories[j].copy()
                        current_log_probs[j], current_log_probs[j+1] = \
                            log_prob_j, log_prob_i
                        swaps.append((i, j, j+1))
            
            # Track base chain
            trajectories.append(current_trajectories[0].copy())
            current_error = np.mean((current_trajectories[0] - self.true_trajectory)**2)
            errors.append(current_error)
            
            # Check convergence
            if current_error < best_error:
                best_error = current_error
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
                
            if (best_error < error_threshold or 
                iterations_without_improvement >= patience):
                break
        
        return (np.array(trajectories), np.array(accepted), 
                np.array(errors), current_trajectories, swaps)

if __name__ == "__main__":
    # Parameters
    steps = 50
    process_noise = 0.1
    motion_noise = 0.05
    n_chains = 8  # Increased number of chains
    max_temp = 100.0  # Higher maximum temperature
    
    # More stringent convergence criteria
    max_iterations = 50000  # Significantly increased from 10000
    error_threshold = 0.001  # Much lower error threshold
    patience = 2000  # Much longer patience
    proposal_std = 0.01  # Smaller steps for more precise convergence
    swap_interval = 2  # More frequent swaps
    
    # Generate data
    true_trajectory = simulate_data(steps, motion_noise)
    
    # Better initialization strategy with more diverse starting points
    initial_trajectories = []
    for i in range(n_chains):
        # Start with straight line between endpoints
        start_point = true_trajectory[0] + np.random.normal(0, 0.5, 2)
        end_point = true_trajectory[-1] + np.random.normal(0, 0.5, 2)
        base_traj = np.linspace(start_point, end_point, steps)
        
        # Add increasing amounts of noise for higher temperature chains
        noise_scale = 0.2 * (i + 1)  # More substantial noise
        noise = np.random.normal(0, noise_scale, (steps, 2))
        # Smooth the noise
        smoothed_noise = np.array([np.convolve(noise[:, j], 
                                              np.ones(5)/5, 
                                              mode='same') 
                                 for j in range(2)]).T
        initial_traj = base_traj + smoothed_noise
        initial_trajectories.append(initial_traj)
    
    # Initialize and run parallel tempering
    pt = ParallelTempering(true_trajectory, process_noise, n_chains, max_temp)
    
    results = pt.run_chain(
        initial_trajectories,
        max_iterations=max_iterations,
        proposal_std=proposal_std,
        error_threshold=error_threshold,
        patience=patience,
        swap_interval=swap_interval
    )
    trajectories, accepted, errors, final_chains, swaps = results
    
    n_iterations = len(trajectories) - 1
    print(f"Chain finished after {n_iterations} iterations")
    print(f"Final MSE: {errors[-1]:.6f}")
    print(f"Acceptance rate: {np.mean(accepted) * 100:.1f}%")
    print(f"Number of successful swaps: {len(swaps)}")
    
    # Setup animation
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])  # Trajectory plot
    ax2 = fig.add_subplot(gs[0, 1])  # Error plot
    ax3 = fig.add_subplot(gs[1, :])  # All chains plot
    
    fig.suptitle('Parallel Tempering Trajectory Estimation', fontsize=16, y=0.95)
    
    # Setup trajectory plot
    padding = 0.5
    x_min, x_max = true_trajectory[:, 0].min(), true_trajectory[:, 0].max()
    y_min, y_max = true_trajectory[:, 1].min(), true_trajectory[:, 1].max()
    max_range = max(x_max - x_min, y_max - y_min)
    center = np.array([(x_max + x_min)/2, (y_max + y_min)/2])
    
    margin = max_range * 0.1
    max_range += margin
    
    for ax in [ax1, ax3]:
        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.grid(True)
    
    # Plot elements
    ax1.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-', 
             label='True trajectory', alpha=0.5)
    current_line, = ax1.plot([], [], 'g-', label='Current', linewidth=2)
    proposed_line, = ax1.plot([], [], 'r--', label='Proposed', linewidth=2)
    
    # Chain lines for all temperatures
    chain_lines = []
    for i in range(n_chains):
        # Alpha decreases exponentially with temperature
        alpha = np.exp(-3 * i / (n_chains - 1))  # Will go from 1.0 to ~0.05
        line, = ax3.plot([], [], '-', 
                        label=f'T={pt.temperatures[i]:.1f}', 
                        alpha=alpha)
        chain_lines.append(line)
    
    # Error plot
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_yscale('log')
    error_line, = ax2.plot([], [], 'b-', label='MSE')
    ax2.grid(True)
    
    # Add info text
    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for ax in [ax1, ax2, ax3]:
        ax.legend(loc='lower right')
    
    def init():
        current_line.set_data([], [])
        proposed_line.set_data([], [])
        error_line.set_data([], [])
        for line in chain_lines:
            line.set_data([], [])
        info_text.set_text('')
        return [current_line, proposed_line, error_line] + chain_lines + [info_text]
    
    def animate(frame):
        if frame < n_iterations:
            # Show current and proposed trajectories (base chain)
            current_traj = trajectories[frame]
            proposed_traj = trajectories[frame + 1]
            
            current_line.set_data(current_traj[:, 0], current_traj[:, 1])
            proposed_line.set_data(proposed_traj[:, 0], proposed_traj[:, 1])
            
            # Update error plot
            error_line.set_data(range(frame + 1), errors[:frame + 1])
            ax2.relim()
            ax2.autoscale_view()
            
            # Show all chains if we're at a swap point
            if frame % swap_interval == 0:
                for i, line in enumerate(chain_lines):
                    line.set_data(final_chains[i][:, 0], final_chains[i][:, 1])
            
            # Update info text
            acceptance_rate = np.mean(accepted[:frame+1]) * 100
            n_swaps = sum(1 for s in swaps if s[0] <= frame)
            status = "Accepted" if accepted[frame] else "Rejected"
            info_text.set_text(f'Iteration: {frame + 1}/{n_iterations}\n'
                             f'Proposal: {status}\n'
                             f'Current MSE: {errors[frame]:.3f}\n'
                             f'Acceptance Rate: {acceptance_rate:.1f}%\n'
                             f'Swaps: {n_swaps}')
            
            # Color code based on acceptance
            proposed_line.set_color('g' if accepted[frame] else 'r')
            
        else:
            # Show final result
            current_line.set_data(trajectories[-1][:, 0], trajectories[-1][:, 1])
            proposed_line.set_data([], [])
            error_line.set_data(range(len(errors)), errors)
            
            # Show final state of all chains
            for i, line in enumerate(chain_lines):
                line.set_data(final_chains[i][:, 0], final_chains[i][:, 1])
            
            acceptance_rate = np.mean(accepted) * 100
            final_error = errors[-1]
            info_text.set_text(f'Final Result\n'
                             f'Total Iterations: {n_iterations}\n'
                             f'Final MSE: {final_error:.3f}\n'
                             f'Acceptance Rate: {acceptance_rate:.1f}%\n'
                             f'Total Swaps: {len(swaps)}')
        
        return [current_line, proposed_line, error_line] + chain_lines + [info_text]
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=n_iterations + 1,
                        interval=100, blit=True, repeat=False)
    
    plt.tight_layout()
    plt.show() 