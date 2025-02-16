"""
Beam search implementation for 2D trajectory estimation.

Key components:
1. Beam: Set of k-best partial trajectories
2. Expansion: Generate candidate next positions
3. Scoring: Evaluate partial trajectories
4. Pruning: Keep only top-k candidates
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from metropolis_hastings import simulate_data  # Reuse trajectory simulation

class BeamSearch:
    def __init__(self, true_trajectory, process_noise, beam_width=10, n_candidates=20):
        self.true_trajectory = true_trajectory
        self.process_noise = process_noise
        self.steps = len(true_trajectory)
        self.beam_width = beam_width
        self.n_candidates = n_candidates
        
    def score_trajectory(self, trajectory, t):
        """Score partial trajectory up to time t."""
        # Match with true trajectory
        match_error = np.sum((trajectory[:t+1] - self.true_trajectory[:t+1])**2)
        match_score = -match_error / (2 * self.process_noise**2)
        
        # Smoothness score
        if t > 0:
            velocities = trajectory[1:t+1] - trajectory[:t]
            smoothness_error = np.sum(velocities**2)
            smoothness_score = -smoothness_error / (2 * self.process_noise**2)
        else:
            smoothness_score = 0
            
        return match_score + smoothness_score
    
    def generate_candidates(self, current_pos, t):
        """Generate candidate next positions."""
        # Use true trajectory to guide candidate generation
        true_direction = self.true_trajectory[t] - current_pos
        
        # Generate candidates around the true direction
        angles = np.linspace(0, 2*np.pi, self.n_candidates)
        magnitudes = np.random.normal(np.linalg.norm(true_direction), 
                                    self.process_noise, 
                                    self.n_candidates)
        
        candidates = current_pos + np.column_stack([
            magnitudes * np.cos(angles),
            magnitudes * np.sin(angles)
        ])
        
        return candidates
    
    def search(self):
        """Perform beam search."""
        # Initialize beam with different starting positions
        initial_positions = self.true_trajectory[0] + np.random.normal(
            0, self.process_noise, (self.beam_width, 2))
        
        beam = [np.zeros((self.steps, 2)) for _ in range(self.beam_width)]
        for i in range(self.beam_width):
            beam[i][0] = initial_positions[i]
        
        scores = [self.score_trajectory(traj, 0) for traj in beam]
        
        # Store all intermediate beams for animation
        all_beams = [beam.copy()]
        best_scores = [max(scores)]
        
        # Expand trajectories step by step
        for t in range(1, self.steps):
            candidates = []
            candidate_scores = []
            
            # Expand each trajectory in beam
            for b, trajectory in enumerate(beam):
                # Generate candidate next positions
                next_positions = self.generate_candidates(trajectory[t-1], t)
                
                for next_pos in next_positions:
                    candidate = trajectory.copy()
                    candidate[t] = next_pos
                    candidates.append(candidate)
                    candidate_scores.append(self.score_trajectory(candidate, t))
            
            # Select top-k candidates
            top_k = np.argsort(candidate_scores)[-self.beam_width:]
            beam = [candidates[i] for i in top_k]
            scores = [candidate_scores[i] for i in top_k]
            
            all_beams.append(beam.copy())
            best_scores.append(max(scores))
        
        return all_beams, best_scores

if __name__ == "__main__":
    # Parameters
    steps = 50
    process_noise = 0.1
    motion_noise = 0.05
    beam_width = 10
    n_candidates = 20
    
    # Generate data
    true_trajectory = simulate_data(steps, motion_noise)
    
    # Run beam search
    bs = BeamSearch(true_trajectory, process_noise, beam_width, n_candidates)
    all_beams, best_scores = bs.search()
    
    # Setup visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Beam Search Trajectory Estimation', fontsize=16, y=0.95)
    
    # Setup trajectory plot
    padding = 0.5
    x_min, x_max = true_trajectory[:, 0].min(), true_trajectory[:, 0].max()
    y_min, y_max = true_trajectory[:, 1].min(), true_trajectory[:, 1].max()
    max_range = max(x_max - x_min, y_max - y_min)
    center = np.array([(x_max + x_min)/2, (y_max + y_min)/2])
    
    margin = max_range * 0.1
    max_range += margin
    
    ax1.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
    ax1.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
    ax1.grid(True)
    
    # Plot true trajectory
    ax1.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-', 
            label='True trajectory', alpha=0.5)
    
    # Initialize beam lines
    beam_lines = []
    for _ in range(beam_width):
        line, = ax1.plot([], [], 'r-', alpha=0.3)
        beam_lines.append(line)
    
    # Best trajectory line
    best_line, = ax1.plot([], [], 'g-', label='Best trajectory', linewidth=2)
    
    # Score plot
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Best Score')
    ax2.grid(True)
    score_line, = ax2.plot([], [], 'b-')
    
    # Add info text
    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.legend(loc='lower right')
    
    def init():
        for line in beam_lines:
            line.set_data([], [])
        best_line.set_data([], [])
        score_line.set_data([], [])
        info_text.set_text('')
        return beam_lines + [best_line, score_line, info_text]
    
    def animate(frame):
        # Get current beam
        current_beam = all_beams[frame]
        
        # Update beam lines
        for i, (line, trajectory) in enumerate(zip(beam_lines, current_beam)):
            line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
        
        # Update best trajectory
        best_trajectory = current_beam[-1]  # Last trajectory has highest score
        best_line.set_data(best_trajectory[:frame+1, 0], 
                          best_trajectory[:frame+1, 1])
        
        # Update score plot
        score_line.set_data(range(frame+1), best_scores[:frame+1])
        ax2.relim()
        ax2.autoscale_view()
        
        # Update info text
        current_error = np.mean((best_trajectory[:frame+1] - 
                               true_trajectory[:frame+1])**2)
        info_text.set_text(f'Step: {frame + 1}/{steps}\n'
                          f'Current MSE: {current_error:.3f}\n'
                          f'Beam Width: {beam_width}')
        
        return beam_lines + [best_line, score_line, info_text]
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=steps,
                        interval=100, blit=True, repeat=False)
    
    plt.tight_layout()
    plt.show() 