"""
Side-by-side comparison of Metropolis-Hastings with and without Parallel Tempering.
Shows how parallel tempering helps escape local optima and explore the space better.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from metropolis_hastings import MetropolisHastings, simulate_data
from parallel_tempering import ParallelTempering

if __name__ == "__main__":
    # Parameters
    steps = 50
    process_noise = 0.1
    motion_noise = 0.05
    
    # MH parameters
    max_iterations = 100000
    error_threshold = 0.0005
    patience = 5000
    proposal_std = 0.01
    
    # PT specific parameters
    n_chains = 8
    max_temp = 100.0
    swap_interval = 2
    
    # Generate data
    true_trajectory = simulate_data(steps, motion_noise)
    
    # Create base initialization with moderate noise
    def create_noisy_trajectory(scale=0.5, rotate=True):
        """Helper to create noisy initialization."""
        # Add smoothed noise
        noise = np.random.normal(0, scale, true_trajectory.shape)
        window = 5
        smoothed_noise = np.array([np.convolve(noise[:, i], 
                                             np.ones(window)/window, 
                                             mode='same') 
                                 for i in range(2)]).T
        
        noisy_traj = true_trajectory + smoothed_noise
        
        if rotate:
            # Add mild rotation to make it more challenging
            angle = np.pi/6  # 30 degrees rotation (reduced from 60)
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
            noisy_traj = 1.2 * (noisy_traj @ rotation)  # Reduced scale from 1.5
        
        return noisy_traj
    
    # Run standard MH with initialization closer to ground truth
    mh = MetropolisHastings(true_trajectory, process_noise)
    initial_trajectory = create_noisy_trajectory(scale=0.2)
    
    mh_results = mh.run_chain(
        initial_trajectory,
        max_iterations=max_iterations,
        proposal_std=proposal_std,
        error_threshold=error_threshold,
        patience=patience
    )
    mh_trajectories, mh_accepted, mh_errors = mh_results
    
    # Run Parallel Tempering with progressive initialization
    pt = ParallelTempering(true_trajectory, process_noise, n_chains, max_temp)
    initial_trajectories = []
    
    # First chain starts close to truth, others progressively more perturbed
    base_scales = np.linspace(0.1, 0.3, n_chains)  # Progressive noise levels (closer to truth)
    for i in range(n_chains):
        noisy_traj = create_noisy_trajectory(
            scale=base_scales[i],
            rotate=(i > 0)  # Only rotate higher temperature chains
        )
        initial_trajectories.append(noisy_traj)
    
    pt_results = pt.run_chain(
        initial_trajectories,
        max_iterations=max_iterations,
        proposal_std=proposal_std,
        error_threshold=error_threshold,
        patience=patience,
        swap_interval=swap_interval
    )
    pt_trajectories, pt_accepted, pt_errors, final_chains, swaps = pt_results
    
    # Print results
    print("Metropolis-Hastings:")
    print(f"Iterations: {len(mh_trajectories)-1}")
    print(f"Final MSE: {mh_errors[-1]:.6f}")
    print(f"Acceptance rate: {np.mean(mh_accepted)*100:.1f}%")
    print("\nParallel Tempering:")
    print(f"Iterations: {len(pt_trajectories)-1}")
    print(f"Final MSE: {pt_errors[-1]:.6f}")
    print(f"Acceptance rate: {np.mean(pt_accepted)*100:.1f}%")
    print(f"Number of swaps: {len(swaps)}")
    
    # Setup visualization
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    ax_mh = fig.add_subplot(gs[0, 0])  # MH trajectory
    ax_pt = fig.add_subplot(gs[0, 1])  # PT trajectory
    ax_error = fig.add_subplot(gs[1, :])  # Error comparison
    
    fig.suptitle('MCMC Comparison: Standard MH vs Parallel Tempering', 
                fontsize=16, y=0.95)
    
    # Setup trajectory plots
    padding = 0.5
    x_min, x_max = true_trajectory[:, 0].min(), true_trajectory[:, 0].max()
    y_min, y_max = true_trajectory[:, 1].min(), true_trajectory[:, 1].max()
    max_range = max(x_max - x_min, y_max - y_min)
    center = np.array([(x_max + x_min)/2, (y_max + y_min)/2])
    
    margin = max_range * 0.1
    max_range += margin
    
    for ax in [ax_mh, ax_pt]:
        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.grid(True)
    
    # Plot elements
    for ax in [ax_mh, ax_pt]:
        ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-', 
               label='True trajectory', alpha=0.5)
    
    mh_line, = ax_mh.plot([], [], 'r-', label='Current', linewidth=2)
    pt_line, = ax_pt.plot([], [], 'g-', label='Current', linewidth=2)
    
    # Error plot
    ax_error.set_xlabel('Iteration')
    ax_error.set_ylabel('Mean Squared Error')
    ax_error.set_yscale('log')
    mh_error_line, = ax_error.plot([], [], 'r-', label='MH Error')
    pt_error_line, = ax_error.plot([], [], 'g-', label='PT Error')
    ax_error.grid(True)
    
    # Add info text
    mh_text = ax_mh.text(0.02, 0.98, '', transform=ax_mh.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    pt_text = ax_pt.text(0.02, 0.98, '', transform=ax_pt.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set titles and legends
    ax_mh.set_title('Standard Metropolis-Hastings', color='red', pad=20)
    ax_pt.set_title('Parallel Tempering', color='green', pad=20)
    for ax in [ax_mh, ax_pt, ax_error]:
        ax.legend(loc='lower right')
    
    def init():
        mh_line.set_data([], [])
        pt_line.set_data([], [])
        mh_error_line.set_data([], [])
        pt_error_line.set_data([], [])
        mh_text.set_text('')
        pt_text.set_text('')
        return mh_line, pt_line, mh_error_line, pt_error_line, mh_text, pt_text
    
    def animate(frame):
        # Get current iteration for each method
        mh_iter = min(frame, len(mh_trajectories)-1)
        pt_iter = min(frame, len(pt_trajectories)-1)
        
        # Update trajectories
        mh_line.set_data(mh_trajectories[mh_iter][:, 0], 
                        mh_trajectories[mh_iter][:, 1])
        pt_line.set_data(pt_trajectories[pt_iter][:, 0], 
                        pt_trajectories[pt_iter][:, 1])
        
        # Update error plots
        mh_error_line.set_data(range(mh_iter+1), mh_errors[:mh_iter+1])
        pt_error_line.set_data(range(pt_iter+1), pt_errors[:pt_iter+1])
        ax_error.relim()
        ax_error.autoscale_view()
        
        # Update info text
        if mh_iter < len(mh_trajectories)-1:
            mh_acc_rate = np.mean(mh_accepted[:mh_iter+1]) * 100
            mh_text.set_text(f'Iteration: {mh_iter + 1}\n'
                           f'Current MSE: {mh_errors[mh_iter]:.3f}\n'
                           f'Acceptance: {mh_acc_rate:.1f}%')
        
        if pt_iter < len(pt_trajectories)-1:
            pt_acc_rate = np.mean(pt_accepted[:pt_iter+1]) * 100
            n_swaps = sum(1 for s in swaps if s[0] <= pt_iter)
            pt_text.set_text(f'Iteration: {pt_iter + 1}\n'
                           f'Current MSE: {pt_errors[pt_iter]:.3f}\n'
                           f'Acceptance: {pt_acc_rate:.1f}%\n'
                           f'Swaps: {n_swaps}')
        
        return mh_line, pt_line, mh_error_line, pt_error_line, mh_text, pt_text
    
    max_frames = max(len(mh_trajectories), len(pt_trajectories))
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=max_frames,
                        interval=50, blit=True, repeat=False)
    
    plt.tight_layout()
    plt.show() 