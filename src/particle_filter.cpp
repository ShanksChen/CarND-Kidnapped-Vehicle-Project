/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  // Gaussian distribution for noise
  std::normal_distribution<double> x_nor(0, std[0]);
  std::normal_distribution<double> y_nor(0, std[1]);
  std::normal_distribution<double> theta_nor(0, std[2]);

  // initialize the particles
  for(int i = 0; i < num_particles; i++)
  {
    Particle p;

    p.id = i;
    p.x = x;
    p.y = y;
    p.theta = theta;
    p.weight = 1.0;

    // add noise
    p.x += x_nor(gen);
    p.y += y_nor(gen);
    p.theta += theta_nor(gen);

    particles.push_back(p);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // Gaussian distribution for noise
  std::normal_distribution<double> x_nor(0, std_pos[0]);
  std::normal_distribution<double> y_nor(0, std_pos[1]);
  std::normal_distribution<double> theta_nor(0, std_pos[2]);
  
  for(int i = 0; i < num_particles; i++)
  {
    if (fabs(yaw_rate) < 0.0000001) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    particles[i].x += x_nor(gen);
    particles[i].y += y_nor(gen);
    particles[i].theta += theta_nor(gen);
  }
  
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase. 
   */
  for(int  i = 0; i < observations.size(); i++)
  {
    LandmarkObs ob = observations[i];

    // initialize the minimum distance to the maximum value
    double min_dist = std::numeric_limits<double>::max();

    // initialize the id of landmark to be associated with the observation
    int map_id = -1;

    for(int j = 0; j < predicted.size(); j++)
    {
      LandmarkObs p = predicted[j];

      double current_dist = dist(ob.x, ob.y, p.x, p.y);
      
      if (current_dist < min_dist) {
        min_dist = current_dist;
        map_id = p.id;
      }      
    }
    // save the nearest id to observation
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for(int i = 0; i < num_particles; i++)
  {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    // new vector to save the landmark location predicted in the sensor range
    vector<LandmarkObs> predictions;

    for(int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;

      // the distance from the landmark and particle
      double sensor_dist = dist(landmark_x, landmark_y, p_x, p_y);

      // the sensor range is larger than the distance from the landmark and particle, save the landmark's x,y and id in the predictions
      if (sensor_dist <= sensor_range) {
        predictions.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }
    }

    // new vector saved the observation which transformed from car coordinates to map coordinates
    vector<LandmarkObs> transformed_observations;

    for(int j = 0; j < observations.size(); j++)
    {
      double trans_x = p_x + cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y;
      double trans_y = p_y + sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y;
      transformed_observations.push_back(LandmarkObs{observations[j].id, trans_x, trans_y});
    }

    dataAssociation(predictions, transformed_observations);

    // initlize the weight again
    particles[i].weight = 1.0;

    for(int k = 0; k < transformed_observations.size(); k++)
    {
      double mu_x = transformed_observations[k].x;
      double mu_y = transformed_observations[k].y;
      double x, y;

      // get the x,y coordinates of the predictions with is same in transformed_observations
      for(int l = 0; l < predictions.size(); l++)
      {
        if (predictions[l].id == transformed_observations[k].id) {
          x = predictions[l].x;
          y = predictions[l].y;
        }
      }
      // calculate the obsrvation weight
      double observation_weight = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1])) * exp(-((pow(x - mu_x, 2) / (2 * std_landmark[0] * std_landmark[0])) + ((pow(y - mu_y, 2) / (2 * std_landmark[1] * std_landmark[1])))));
      // save the product of this observation with total weight
      particles[i].weight *= observation_weight; 
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles;

  vector<double> weights;

  for(int i = 0; i < num_particles; i++)
  {
    weights.push_back(particles[i].weight);
  }

  std::uniform_int_distribution<int> uni_int_dist(0, num_particles - 1);
  auto index = uni_int_dist(gen);

  double max_weight = *max_element(weights.begin(), weights.end());

  std::uniform_real_distribution<double> uni_real_dist(0, max_weight);

  double beta = 0.0;

  for(int i = 0; i < num_particles; i++)
  {
    beta += uni_real_dist(gen) * 2;

    while(beta > weights[index]){
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }

    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}