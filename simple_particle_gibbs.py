"""
Simple particle Gibbs implementation using the particle filter.
"""

import numpy as np
from simple_particle_filter import SimpleParticleFilter

class SimpleParticleGibbs:
    def __init__(self, num_particles, space_limits, process_noise, measurement_noise):
        self.pf = SimpleParticleFilter(
            num_particles=num_particles,
            space_limits=space_limits,
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )
        self.current_trajectory = None
        
    def conditional_particle_filter(self, measurements, conditioned_trajectory):
        """Run particle filter while conditioning on a trajectory."""
        T = len(measurements)
        self.pf.initialize()
        
        # Fix one particle to the conditioned trajectory
        conditioned_idx = 0
        self.pf.particles[conditioned_idx] = conditioned_trajectory[0]
        trajectory = np.zeros((T, 2))
        
        for t in range(T):
            if t > 0:
                # Predict all except conditioned particle
                self.pf.predict()
                self.pf.particles[conditioned_idx] = conditioned_trajectory[t]
            
            self.pf.update(measurements[t])
            trajectory[t] = self.pf.get_estimate()
            
            if t < T - 1:
                # Conditional resampling: keep conditioned particle
                indices = np.random.choice(
                    self.pf.num_particles,
                    size=self.pf.num_particles - 1,
                    p=self.pf.weights
                )
                indices = np.insert(indices, conditioned_idx, conditioned_idx)
                
                self.pf.particles = self.pf.particles[indices]
                self.pf.weights = np.ones(self.pf.num_particles) / self.pf.num_particles
        
        return trajectory
    
    def run_iteration(self, measurements):
        """Run one iteration of Particle Gibbs."""
        if self.current_trajectory is None:
            # First iteration: run regular particle filter
            self.current_trajectory = self.pf.run_filter(measurements)
        else:
            # Subsequent iterations: run conditional particle filter
            self.current_trajectory = self.conditional_particle_filter(
                measurements, 
                self.current_trajectory
            )
        return self.current_trajectory.copy()

    def run_iteration_with_history(self, measurements, conditioned_trajectory=None):
        """Run particle filter while conditioning on a trajectory and store history."""
        T = len(measurements)
        self.pf.initialize()
        
        # Fix one particle to the conditioned trajectory
        conditioned_idx = 0
        if conditioned_trajectory is not None:
            self.pf.particles[conditioned_idx] = conditioned_trajectory[0]
        
        trajectory = np.zeros((T, 2))
        particles_history = []
        weights_history = []
        
        for t in range(T):
            if t > 0:
                # Predict all except conditioned particle
                self.pf.predict()
                if conditioned_trajectory is not None:
                    self.pf.particles[conditioned_idx] = conditioned_trajectory[t]
            
            self.pf.update(measurements[t])
            trajectory[t] = self.pf.get_estimate()
            
            # Store current state
            particles_history.append(self.pf.particles.copy())
            weights_history.append(self.pf.weights.copy())
            
            if t < T - 1:
                # Conditional resampling
                indices = np.random.choice(
                    self.pf.num_particles,
                    size=self.pf.num_particles - 1,
                    p=self.pf.weights
                )
                if conditioned_trajectory is not None:
                    indices = np.insert(indices, conditioned_idx, conditioned_idx)
                
                self.pf.particles = self.pf.particles[indices]
                self.pf.weights = np.ones(self.pf.num_particles) / self.pf.num_particles
        
        return trajectory, particles_history, weights_history 