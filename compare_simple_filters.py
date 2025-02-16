"""
Animated comparison of simple particle filter vs particle Gibbs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simple_particle_filter import SimpleParticleFilter
from simple_particle_gibbs import SimpleParticleGibbs

def generate_data(steps):
    """Generate ground truth trajectory and noisy measurements."""
    # Random initial position
    initial_pos = np.random.uniform(-2, 2, 2)
    
    # Generate interesting spiral motion with random variations
    t = np.linspace(0, 4*np.pi, steps)
    true_trajectory = np.zeros((steps, 2))
    true_trajectory[0] = initial_pos
    
    # Add spiral motion with random variations
    radius = 0.2 * t * (1 + 0.2 * np.random.randn())  # Random scale
    phase = 2 * np.pi * np.random.rand()  # Random phase
    freq = 1 + 0.2 * np.random.randn()  # Random frequency
    
    # Generate base spiral
    true_trajectory[:, 0] = initial_pos[0] + radius * np.cos(freq * t + phase)
    true_trajectory[:, 1] = initial_pos[1] + radius * np.sin(freq * t + phase)
    
    # Add random walk component
    random_walk = np.cumsum(
        np.random.normal(0, 0.1, size=true_trajectory.shape), 
        axis=0
    )
    true_trajectory += random_walk
    
    # Generate noisy measurements
    noise = 0.5 * (1 + 0.2 * np.random.rand())  # Random noise level
    measurements = true_trajectory + np.random.normal(
        0, noise, true_trajectory.shape
    )
    
    return true_trajectory, measurements

def main():
    # Parameters
    num_particles = 100
    space_limits = [-5, 5, -5, 5]  # Smaller limits for better zoom
    process_noise = 0.2
    measurement_noise = 0.5
    steps = 100
    pg_iterations = 10
    
    # Generate data
    true_trajectory, measurements = generate_data(steps)
    
    # Initialize algorithms
    pf = SimpleParticleFilter(
        num_particles, space_limits, process_noise, measurement_noise
    )
    pg = SimpleParticleGibbs(
        num_particles, space_limits, process_noise, measurement_noise
    )
    
    # Run particle filter first
    pf_trajectory = pf.run_filter(measurements)
    
    # For PG, we'll store both final trajectories and intermediate particles
    pg_trajectories = []
    pg_particle_history = []  # Store particles for each step of each iteration
    pg_weight_history = []    # Store weights for visualization
    
    for _ in range(pg_iterations):
        # Run one iteration and store intermediate states
        particles_per_step = []
        weights_per_step = []
        
        # Run conditional particle filter and store intermediate states
        if len(pg_trajectories) == 0:
            # First iteration: run regular particle filter
            pg.pf.initialize()
            trajectory = np.zeros((len(measurements), 2))
            
            for t, measurement in enumerate(measurements):
                if t > 0:
                    pg.pf.predict()
                pg.pf.update(measurement)
                trajectory[t] = pg.pf.get_estimate()
                
                # Store current state
                particles_per_step.append(pg.pf.particles.copy())
                weights_per_step.append(pg.pf.weights.copy())
                
                pg.pf.resample()
                
        else:
            # Subsequent iterations: run conditional particle filter
            trajectory, particles_per_step, weights_per_step = \
                pg.run_iteration_with_history(measurements, pg_trajectories[-1])
        
        pg_trajectories.append(trajectory)
        pg_particle_history.append(particles_per_step)
        pg_weight_history.append(weights_per_step)
    
    # Setup animation with 4 panels (2x2 grid)
    fig = plt.figure(figsize=(20, 14))
    gs = plt.GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])  # Particle Filter
    ax2 = fig.add_subplot(gs[0, 1])  # Current PG iteration
    ax3 = fig.add_subplot(gs[1, 0])  # Evolution of trajectories
    ax4 = fig.add_subplot(gs[1, 1])  # Ancestor sampling visualization
    
    fig.suptitle('Particle Filter vs Particle Gibbs Comparison', y=0.95)
    
    # Calculate plot limits with padding
    padding = 0.5
    x_min = min(true_trajectory[:, 0].min(), measurements[:, 0].min()) - padding
    x_max = max(true_trajectory[:, 0].max(), measurements[:, 0].max()) + padding
    y_min = min(true_trajectory[:, 1].min(), measurements[:, 1].min()) - padding
    y_max = max(true_trajectory[:, 1].max(), measurements[:, 1].max()) + padding
    
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True)
        
    ax1.set_title('Particle Filter\n(Single Pass)')
    ax2.set_title('Particle Gibbs\n(Current Iteration)')
    ax3.set_title('Particle Gibbs\n(All Iterations)')
    ax4.set_title('Ancestor Sampling\n(Segment Colors = Different Ancestors)')
    
    # Remove the particle index axis labels since we're not using them anymore
    ax4.set_xlabel('X position')
    ax4.set_ylabel('Y position')
    
    # Plot elements
    pf_lines = {
        'true': ax1.plot([], [], 'b-', alpha=0.5, label='True')[0],
        'meas': ax1.plot([], [], 'kx', alpha=0.3, label='Measurements')[0],
        'est': ax1.plot([], [], 'r-', label='Estimate')[0],
        'particles': [ax1.plot([], [], 'g.', alpha=0.1, markersize=1)[0] 
                     for _ in range(num_particles)]
    }
    
    pg_lines = {
        'true': ax2.plot([], [], 'b-', alpha=0.5, label='True')[0],
        'meas': ax2.plot([], [], 'kx', alpha=0.3, label='Measurements')[0],
        'est': ax2.plot([], [], 'r-', linewidth=2, label='Current')[0],
        'prev': ax2.plot([], [], 'orange', alpha=0.3, label='Previous')[0],
        'particles': [ax2.plot([], [], 'g.', alpha=0.1, markersize=1)[0] 
                     for _ in range(num_particles)]
    }
    
    # New lines for the evolution panel
    evolution_lines = {
        'true': ax3.plot([], [], 'b-', alpha=0.5, label='True')[0],
        'meas': ax3.plot([], [], 'kx', alpha=0.3, label='Measurements')[0],
        'trajectories': [ax3.plot([], [], alpha=0.2)[0] 
                        for _ in range(pg_iterations)],
        'current': ax3.plot([], [], 'r-', linewidth=2, label='Current')[0]
    }
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    
    # Add new plot elements for ancestor sampling visualization
    ancestor_lines = {
        'segments': [ax4.plot([], [], '-', alpha=0.8, linewidth=2)[0] 
                    for _ in range(steps-1)],  # One segment between each pair of points
    }
    
    ax4.legend()
    
    def init():
        # Initialize all lines empty
        for line in pf_lines['particles']:
            line.set_data([], [])
        for line in pg_lines['particles']:
            line.set_data([], [])
        for k in ['true', 'meas', 'est', 'prev']:
            if k in pf_lines:
                pf_lines[k].set_data([], [])
            if k in pg_lines:
                pg_lines[k].set_data([], [])
        
        # Initialize evolution lines
        for line in evolution_lines['trajectories']:
            line.set_data([], [])
        evolution_lines['true'].set_data([], [])
        evolution_lines['meas'].set_data([], [])
        evolution_lines['current'].set_data([], [])
        
        # Initialize ancestor sampling lines
        for line in ancestor_lines['segments']:
            line.set_data([], [])
        
        # Collect all lines for animation
        lines = []
        # Add particle filter lines
        lines.extend([pf_lines[k] for k in ['true', 'meas', 'est'] if k in pf_lines])
        lines.extend(pf_lines['particles'])
        
        # Add particle gibbs lines
        lines.extend([pg_lines[k] for k in ['true', 'meas', 'est', 'prev'] if k in pg_lines])
        lines.extend(pg_lines['particles'])
        
        # Add evolution lines
        lines.append(evolution_lines['true'])
        lines.append(evolution_lines['meas'])
        lines.extend(evolution_lines['trajectories'])
        lines.append(evolution_lines['current'])
        
        # Add ancestor sampling lines
        lines.extend(ancestor_lines['segments'])
        
        return lines
    
    def animate(frame):
        pg_iter = frame // steps
        t = frame % steps
        
        # Update particle filter plot (left panel)
        pf_lines['true'].set_data(true_trajectory[:t+1, 0], true_trajectory[:t+1, 1])
        pf_lines['meas'].set_data(measurements[:t+1, 0], measurements[:t+1, 1])
        pf_lines['est'].set_data(pf_trajectory[:t+1, 0], pf_trajectory[:t+1, 1])
        
        # Show current particles for PF
        particles = pf.particles
        for i, line in enumerate(pf_lines['particles']):
            if particles is not None and i < len(particles):
                line.set_data([particles[i, 0]], [particles[i, 1]])
        
        # Update particle Gibbs plot (middle panel)
        pg_lines['true'].set_data(true_trajectory[:t+1, 0], true_trajectory[:t+1, 1])
        pg_lines['meas'].set_data(measurements[:t+1, 0], measurements[:t+1, 1])
        
        # Show previous iteration's full trajectory
        if pg_iter > 0:
            prev_traj = pg_trajectories[pg_iter-1]
            pg_lines['prev'].set_data(prev_traj[:t+1, 0], prev_traj[:t+1, 1])
        
        # Show current particles and partial trajectory
        current_particles = pg_particle_history[pg_iter][t]
        current_weights = pg_weight_history[pg_iter][t]
        
        # Update particles with size proportional to weight
        max_size = 50
        sizes = max_size * current_weights / current_weights.max()
        
        for i, (line, size) in enumerate(zip(pg_lines['particles'], sizes)):
            line.set_data([current_particles[i, 0]], [current_particles[i, 1]])
            line.set_markersize(np.sqrt(size))
        
        # Show current partial trajectory
        current_traj = pg_trajectories[pg_iter]
        pg_lines['est'].set_data(current_traj[:t+1, 0], current_traj[:t+1, 1])
        
        # Update evolution plot (right panel)
        evolution_lines['true'].set_data(true_trajectory[:, 0], true_trajectory[:, 1])
        evolution_lines['meas'].set_data(measurements[:, 0], measurements[:, 1])
        
        # Show all previous trajectories with color gradient
        for i in range(pg_iter):
            traj = pg_trajectories[i]
            line = evolution_lines['trajectories'][i]
            # Color gradient from light blue to orange
            color = plt.cm.RdYlBu(i / pg_iterations)
            line.set_color(color)
            line.set_data(traj[:, 0], traj[:, 1])
            
        # Clear future trajectories
        for i in range(pg_iter, pg_iterations):
            evolution_lines['trajectories'][i].set_data([], [])
            
        # Show current trajectory
        evolution_lines['current'].set_data(
            current_traj[:t+1, 0], 
            current_traj[:t+1, 1]
        )
        
        # Update titles
        ax2.set_title(
            f'Particle Gibbs\nIteration {pg_iter + 1}/{pg_iterations}\n'
            f'Step {t + 1}/{steps}'
        )
        ax3.set_title(
            f'Evolution of Trajectories\n'
            f'Showing iterations 1-{pg_iter + 1}'
        )
        
        # Update ancestor sampling visualization
        if pg_iter > 0:  # Only show after first iteration
            current_traj = pg_trajectories[pg_iter]
            
            # For each time step, show segment in a different color if it comes
            # from a different ancestor
            for i in range(min(t, steps-1)):
                segment = ancestor_lines['segments'][i]
                
                # Sample ancestor for this segment
                ancestor_idx = np.random.choice(
                    num_particles,
                    p=pg_weight_history[pg_iter-1][i]
                )
                
                # Use color based on ancestor index
                color = plt.cm.tab20(ancestor_idx / num_particles)
                segment.set_color(color)
                
                # Plot segment between consecutive points
                segment.set_data(
                    [current_traj[i, 0], current_traj[i+1, 0]],
                    [current_traj[i, 1], current_traj[i+1, 1]]
                )
            
            # Clear unused segments
            for i in range(t, steps-1):
                ancestor_lines['segments'][i].set_data([], [])
            
            ax4.set_title(
                f'Ancestor Sampling\n'
                f'Colors show segments from different ancestors'
            )
        
        # Collect all lines in the same order as init
        lines = []
        # Add particle filter lines
        lines.extend([pf_lines[k] for k in ['true', 'meas', 'est'] if k in pf_lines])
        lines.extend(pf_lines['particles'])
        
        # Add particle gibbs lines
        lines.extend([pg_lines[k] for k in ['true', 'meas', 'est', 'prev'] if k in pg_lines])
        lines.extend(pg_lines['particles'])
        
        # Add evolution lines
        lines.append(evolution_lines['true'])
        lines.append(evolution_lines['meas'])
        lines.extend(evolution_lines['trajectories'])
        lines.append(evolution_lines['current'])
        
        # Add ancestor sampling lines
        lines.extend(ancestor_lines['segments'])
        
        return lines
    
    anim = FuncAnimation(
        fig, animate, init_func=init, 
        frames=steps * pg_iterations,
        interval=50, blit=True
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 