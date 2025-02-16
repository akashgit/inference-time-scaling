"""
Metropolis-Hastings implementation for 2D trajectory estimation.

Key components:
1. Proposal distribution: Perturbs the current trajectory
2. Likelihood: How well trajectory matches measurements
3. Accept/Reject: Based on posterior ratio
4. Animation: Shows proposal and acceptance/rejection process
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

def simulate_data(steps, motion_noise):
    """Simulate ground truth trajectory with smoothing."""
    # Generate smoother trajectory using cumulative sum of velocities
    velocities = np.random.normal(0, motion_noise, (steps, 2))
    # Apply smoothing to velocities
    window = 5  # Smoothing window
    velocities = np.array([np.convolve(velocities[:, i], np.ones(window)/window, mode='same') 
                          for i in range(2)]).T
    
    # Generate trajectory from velocities
    true_trajectory = np.cumsum(velocities, axis=0)
    
    # Center the trajectory
    true_trajectory -= np.mean(true_trajectory, axis=0)
    
    return true_trajectory

class MetropolisHastings:
    def __init__(self, true_trajectory, process_noise):
        self.true_trajectory = true_trajectory
        self.process_noise = process_noise
        self.steps = len(true_trajectory)
        
    def log_likelihood(self, trajectory):
        """Compute log likelihood under the target model."""
        # Direct comparison with true trajectory
        trajectory_error = np.sum((trajectory - self.true_trajectory)**2)
        log_target = -trajectory_error / (2 * self.process_noise**2)
        
        # Process likelihood (smoothness prior)
        if self.steps > 1:
            velocities = trajectory[1:] - trajectory[:-1]
            process_error = np.sum(velocities**2)
            log_process = -process_error / (2 * self.process_noise**2)
        else:
            log_process = 0
            
        return log_target + log_process
    
    def propose_trajectory(self, current_trajectory, proposal_std=0.1):
        """Propose new trajectory by adding noise to current one."""
        # Add random perturbations to the trajectory
        noise = np.random.normal(0, proposal_std, current_trajectory.shape)
        
        # Make perturbations correlated in time for smoother proposals
        noise = np.cumsum(noise, axis=0) * 0.1
        
        return current_trajectory + noise
    
    def run_chain(self, initial_trajectory, max_iterations=1000, proposal_std=0.1,
                 error_threshold=0.1, patience=50):
        """
        Run the MCMC chain until convergence or max iterations.
        
        Args:
            error_threshold: Target mean squared error per timestep
            patience: Number of iterations to wait for improvement
        """
        current_trajectory = initial_trajectory.copy()
        current_log_prob = self.log_likelihood(current_trajectory)
        
        trajectories = [current_trajectory.copy()]
        accepted = []
        errors = []
        
        # Track best trajectory and its error
        best_error = float('inf')
        best_trajectory = current_trajectory.copy()
        iterations_without_improvement = 0
        
        for i in range(max_iterations):
            # Propose and evaluate
            proposed_trajectory = self.propose_trajectory(current_trajectory, proposal_std)
            proposed_log_prob = self.log_likelihood(proposed_trajectory)
            
            # Accept/reject
            log_ratio = proposed_log_prob - current_log_prob
            if np.log(np.random.random()) < log_ratio:
                current_trajectory = proposed_trajectory
                current_log_prob = proposed_log_prob
                accepted.append(True)
            else:
                accepted.append(False)
            
            # Calculate error
            current_error = np.mean((current_trajectory - self.true_trajectory)**2)
            errors.append(current_error)
            
            # Track best trajectory
            if current_error < best_error:
                best_error = current_error
                best_trajectory = current_trajectory.copy()
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
            
            trajectories.append(current_trajectory.copy())
            
            # Check convergence
            if (best_error < error_threshold or 
                iterations_without_improvement >= patience):
                break
        
        return np.array(trajectories), np.array(accepted), np.array(errors)

if __name__ == "__main__":
    # Parameters
    steps = 50
    process_noise = 0.1  # Reduced from 0.2 for smoother trajectories
    motion_noise = 0.05  # Reduced from 0.1 for smoother trajectories
    
    # More stringent convergence criteria
    max_iterations = 5000
    error_threshold = 0.01
    patience = 200
    proposal_std = 0.02  # Reduced for finer exploration
    
    # Generate data (just true trajectory)
    true_trajectory = simulate_data(steps, motion_noise)
    
    # Initialize and run MCMC
    mh = MetropolisHastings(true_trajectory, process_noise)
    initial_trajectory = simulate_data(steps, motion_noise)
    trajectories, accepted, errors = mh.run_chain(
        initial_trajectory,
        max_iterations=max_iterations,
        proposal_std=proposal_std,
        error_threshold=error_threshold,
        patience=patience
    )
    
    n_iterations = len(trajectories) - 1
    print(f"Chain finished after {n_iterations} iterations")
    print(f"Final MSE: {errors[-1]:.6f}")
    print(f"Acceptance rate: {np.mean(accepted) * 100:.1f}%")
    
    # Setup animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Metropolis-Hastings Trajectory Estimation', fontsize=16, y=0.95)
    
    # Setup trajectory plot
    padding = 0.5  # Reduced from 2 for tighter zoom
    x_min, x_max = true_trajectory[:, 0].min(), true_trajectory[:, 0].max()
    y_min, y_max = true_trajectory[:, 1].min(), true_trajectory[:, 1].max()
    max_range = max(x_max - x_min, y_max - y_min)
    center = np.array([(x_max + x_min)/2, (y_max + y_min)/2])
    
    # Add small margin to ensure trajectory is fully visible
    margin = max_range * 0.1
    max_range += margin
    
    ax1.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
    ax1.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
    
    # Plot true trajectory (no measurements)
    ax1.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-', 
            label='True trajectory', alpha=0.5)
    
    # Initialize plot elements
    current_line, = ax1.plot([], [], 'g-', label='Current', linewidth=2)
    proposed_line, = ax1.plot([], [], 'r--', label='Proposed', linewidth=2)
    
    # Setup error plot
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_yscale('log')
    error_line, = ax2.plot([], [], 'b-', label='MSE')
    ax2.grid(True)
    
    # Add info text
    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.grid(True)
    ax1.legend(loc='lower right')
    ax2.legend(loc='upper right')
    
    def init():
        current_line.set_data([], [])
        proposed_line.set_data([], [])
        error_line.set_data([], [])
        info_text.set_text('')
        return current_line, proposed_line, error_line, info_text
    
    def animate(frame):
        if frame < n_iterations:
            # Show current and proposed trajectories
            current_traj = trajectories[frame]
            proposed_traj = trajectories[frame + 1]
            
            current_line.set_data(current_traj[:, 0], current_traj[:, 1])
            proposed_line.set_data(proposed_traj[:, 0], proposed_traj[:, 1])
            
            # Update error plot
            error_line.set_data(range(frame + 1), errors[:frame + 1])
            ax2.relim()
            ax2.autoscale_view()
            
            # Update info text
            acceptance_rate = np.mean(accepted[:frame+1]) * 100
            status = "Accepted" if accepted[frame] else "Rejected"
            info_text.set_text(f'Iteration: {frame + 1}/{n_iterations}\n'
                             f'Proposal: {status}\n'
                             f'Current MSE: {errors[frame]:.3f}\n'
                             f'Acceptance Rate: {acceptance_rate:.1f}%')
            
            # Color code based on acceptance
            proposed_line.set_color('g' if accepted[frame] else 'r')
            
        else:
            # Show final result
            current_line.set_data(trajectories[-1][:, 0], trajectories[-1][:, 1])
            proposed_line.set_data([], [])
            error_line.set_data(range(len(errors)), errors)
            
            acceptance_rate = np.mean(accepted) * 100
            final_error = errors[-1]
            info_text.set_text(f'Final Result\n'
                             f'Total Iterations: {n_iterations}\n'
                             f'Final MSE: {final_error:.3f}\n'
                             f'Acceptance Rate: {acceptance_rate:.1f}%')
        
        return current_line, proposed_line, error_line, info_text
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=n_iterations + 1,
                        interval=100, blit=True, repeat=False)
    
    plt.tight_layout()
    plt.show() 