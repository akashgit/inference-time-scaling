"""
Simple particle filter implementation for 2D object tracking.
"""

import numpy as np

class SimpleParticleFilter:
    def __init__(self, num_particles, space_limits, process_noise, measurement_noise):
        self.num_particles = num_particles
        self.space_limits = space_limits  # [x_min, x_max, y_min, y_max]
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Initialize storage
        self.particles = None
        self.weights = None
        
    def initialize(self):
        """Initialize particles uniformly in the state space."""
        self.particles = np.random.uniform(
            low=[self.space_limits[0], self.space_limits[2]],
            high=[self.space_limits[1], self.space_limits[3]],
            size=(self.num_particles, 2)
        )
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def predict(self):
        """Propagate particles through motion model."""
        self.particles += np.random.normal(
            0, self.process_noise, 
            size=self.particles.shape
        )
        
    def update(self, measurement):
        """Update weights based on measurement."""
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        self.weights = np.exp(-distances**2 / (2 * self.measurement_noise**2))
        self.weights /= np.sum(self.weights)
        
    def resample(self):
        """Resample particles based on weights."""
        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def get_estimate(self):
        """Get current state estimate (weighted mean of particles)."""
        return np.average(self.particles, weights=self.weights, axis=0)
        
    def run_filter(self, measurements):
        """Run particle filter for a sequence of measurements."""
        self.initialize()
        trajectory = np.zeros((len(measurements), 2))
        
        for t, measurement in enumerate(measurements):
            if t > 0:
                self.predict()
            self.update(measurement)
            trajectory[t] = self.get_estimate()
            self.resample()
            
        return trajectory 