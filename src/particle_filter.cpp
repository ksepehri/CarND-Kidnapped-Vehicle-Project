/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    num_particles = 100;
    
    default_random_engine gen;
    normal_distribution<double> N_x(x, std[0]);
    normal_distribution<double> N_y(y, std[1]);
    normal_distribution<double> N_theta(theta, std[2]);
    weights = vector<double>(num_particles);
    particles = vector<Particle>(num_particles);
    
    for(int i = 0; i < num_particles; i++) {
        Particle particle;
        particle.id = i;
        particle.x = N_x(gen);
        particle.y = N_y(gen);
        particle.theta = N_theta(gen);
        particle.weight = 1.0f;
        
        particles[i] = particle;
        weights[i] = 1.0f;
        
//        cout << particles[i].id << ", " << weights[i] << endl;

    }
    
    is_initialized = true;
    
    

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find normal_distribution and default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    default_random_engine gen;
    
    
    for(int i = 0; i < num_particles; i++) {
        double new_x, new_y, new_theta;
        
        Particle p = particles[i];
        double tyd = p.theta + yaw_rate * delta_t;
        
        // from last lesson
        if(yaw_rate == 0) {
            new_x = p.x + velocity*delta_t*cos(p.theta);
            new_y = p.y + velocity*delta_t*sin(p.theta);
            new_theta = p.theta;
        }
        else {
            new_x = p.x + (velocity/yaw_rate) * (sin(tyd) - sin(p.theta));
            new_y = p.y + (velocity/yaw_rate) * (cos(p.theta) - cos(tyd));
            new_theta = tyd;
        }
        
        normal_distribution<double> N_x(new_x, std_pos[0]);
        normal_distribution<double> N_y(new_y, std_pos[1]);
        normal_distribution<double> N_theta(new_theta, std_pos[2]);
        
        p.x = N_x(gen);
        p.y = N_y(gen);
        p.theta = N_theta(gen);
        
        particles[i] = p;
        
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    for(int i = 0; i < observations.size(); i++) {
        LandmarkObs obs = observations[i];
        
        double min = numeric_limits<double>::infinity();
        
        int index = -1;
        for(int j = 0; j < predicted.size(); j++){
            LandmarkObs pred = predicted[j];

            double distance = dist(obs.x, obs.y, pred.x, pred.y);
            
            if (distance < min)
            {
                min = distance;
                index = j;
                
            }
        }
        
        observations[i].id = predicted[index].id;
        
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    //from lesson 14.16
    
    for (int i=0; i< num_particles; i++ ){
        Particle p = particles[i];
        
        //convert to map coords
        vector<LandmarkObs> transformed_observations;
        for (int j= 0 ; j < observations.size();j++){
            LandmarkObs o = observations[j];
            LandmarkObs temp_o;
            
            temp_o.id = o.id;
            temp_o.x = p.x + o.x * cos(p.theta) - o.y * sin(p.theta);
            temp_o.y = p.y + o.x * sin(p.theta) + o.y * cos(p.theta);
            
            transformed_observations.push_back(temp_o);
        }
        
        //for all landmarks
        vector<LandmarkObs> predictions;
        for ( int k=0; k< map_landmarks.landmark_list.size(); ++k){
            auto lm = map_landmarks.landmark_list[k];
            float dx = lm.x_f - p.x;
            float dy = lm.y_f - p.y;
            float dist = sqrt(dx*dx + dy*dy);
            
            // Add only if in range
            if(dist < sensor_range){
                LandmarkObs pred;
                pred.x = lm.x_f;
                pred.y = lm.y_f;
                pred.id = lm.id_i;
                
                predictions.push_back(pred);
                
            }
        }
        
        
        dataAssociation(predictions, transformed_observations);
        
        
        particles[i].weight = 1.0;
        
        for(int k = 0; k < transformed_observations.size(); k++)
        {
            double to_x = transformed_observations[k].x;
            double to_y = transformed_observations[k].y;
            double lm_x, lm_y;
            int lm_id;
            
            for ( int k_i = 0 ; k_i <  predictions.size();k_i++)
            {
                if (transformed_observations[k].id == predictions[k_i].id){
                    lm_id = predictions[k_i].id;
                    lm_x = predictions[k_i].x;
                    lm_y = predictions[k_i].y;
                }
                
            }
            
            // calculate weight with Gaussian
            double s_x = std_landmark[0];
            double s_y = std_landmark[1];
            double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(lm_x-to_x,2)/(2*pow(s_x, 2)) + (pow(lm_y-to_y,2)/(2*pow(s_y, 2))) ) );
            
            particles[i].weight *= obs_w;
            
        }
        
        weights[i] = particles[i].weight;
        
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    default_random_engine gen;
    discrete_distribution<int> distribution(weights.begin(),weights.end());
    
    vector<Particle> resample_particles;
    
    for(int i = 0; i < num_particles; i++) {
        resample_particles.push_back(particles[distribution(gen)]);
    }
    
    particles = resample_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const vector<int>& associations, 
                                     const vector<double>& sense_x, const vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
