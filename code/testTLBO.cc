extern "C" {
#include "cec17.h"
}
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>

using namespace std;


/**
 * Initializes the population with random values within the specified bounds.
 *
 * @param population The population to be initialized.
 * @param fitnesses The fitness values of the population.
 * @param dim The dimension of each individual in the population.
 * @param lower_bound The lower bound for the random values.
 * @param upper_bound The upper bound for the random values.
 * @param gen The random number generator.
 */
template <typename Random>
void initializePopulation(vector<vector<double>>& population, vector<double>& fitnesses, int dim, double lower_bound, double upper_bound, Random& gen);

/**
 * Calculates the fitness value for a given solution.
 *
 * @param solution The solution for which the fitness value is calculated.
 * @return The fitness value of the solution.
 */
double calculateFitness(const vector<double>& solution);

/**
 * Performs the teaching phase of the Teaching-Learning-Based Optimization (TLBO) algorithm.
 * 
 * @param population The population of solutions.
 * @param fitnesses The fitness values of the solutions.
 * @param best_solution The best solution found so far.
 * @param best_fitness The fitness value of the best solution found so far.
 * @param dim The dimensionality of the solutions.
 * @param lower_bound The lower bound of the solution space.
 * @param upper_bound The upper bound of the solution space.
 * @param gen The random number generator.
 * @param evals The number of evaluations performed.
 * @param max_evals The maximum number of evaluations allowed.
 */
template <typename Random>
void teachingPhase(vector<vector<double>>& population, vector<double>& fitnesses, vector<double>& best_solution, double& best_fitness, int dim, double lower_bound, double upper_bound, Random& gen, int& evals, int max_evals);

/**
 * Performs the learning phase of the Teaching-Learning-Based Optimization (TLBO) algorithm.
 *
 * This function updates the population by applying the teaching and learning phases of the TLBO algorithm.
 * The teaching phase involves selecting a teacher and updating the population based on the teacher's knowledge.
 * The learning phase involves selecting a learner and updating the population based on the learner's knowledge.
 *
 * @param population   The population of solutions.
 * @param fitnesses    The fitness values of the solutions in the population.
 * @param dim          The dimensionality of the solutions.
 * @param lower_bound  The lower bound of the solution space.
 * @param upper_bound  The upper bound of the solution space.
 * @param gen          The random number generator.
 * @param evals        The current number of evaluations.
 * @param max_evals    The maximum number of evaluations allowed.
 */
template <typename Random>
void learningPhase(vector<vector<double>>& population, vector<double>& fitnesses, int dim, double lower_bound, double upper_bound, Random& gen, int& evals, int max_evals);

/**
 * Implements the Teaching-Learning-Based Optimization (TLBO) algorithm.
 *
 * @param population The population of solutions.
 * @param fitnesses The fitness values of the solutions.
 * @param dim The dimensionality of the solutions.
 * @param lower_bound The lower bound for the solution values.
 * @param upper_bound The upper bound for the solution values.
 * @param gen The random number generator.
 * @param max_evals The maximum number of fitness evaluations.
 */
template <typename Random>
void tlbo(vector<vector<double>>& population, vector<double>& fitnesses, int dim, double lower_bound, double upper_bound, Random& gen, int max_evals);

/**
 * Clips the values of the solution to the specified bounds.
 *
 * @param sol The solution to be clipped.
 * @param lower The lower bound for the solution values.
 * @param upper The upper bound for the solution values.
 */
void clip(vector<double>& sol, double lower, double upper) {
    for (auto& val : sol) {
        if (val < lower) {
            val = lower;
        } else if (val > upper) {
            val = upper;
        }
    }
}

int main() {
    int dim = 50;
    const double lower_bound = -100.0, upper_bound = 100.0;
    const int num_learners = 30;
    const int max_evals = 10000 * dim;

    random_device rd;
    mt19937 gen(rd());

    for (int funcid = 1; funcid <= 30; funcid++) {
        vector<vector<double>> population(num_learners, vector<double>(dim));
        vector<double> fitnesses(num_learners);

        cec17_init("tlbo", funcid, dim);
        initializePopulation(population, fitnesses, dim, lower_bound, upper_bound, gen);
        tlbo(population, fitnesses, dim, lower_bound, upper_bound, gen, max_evals);

        double best_fitness = *min_element(fitnesses.begin(), fitnesses.end());
        cout << "Best Random[F" << funcid << "]: " << scientific << cec17_error(best_fitness) << endl;
    }

    return 0;
}

template <typename Random>
void initializePopulation(vector<vector<double>>& population, vector<double>& fitnesses, int dim, double lower_bound, double upper_bound, Random& gen) {
    uniform_real_distribution<> dis(lower_bound, upper_bound);
    for (auto& individual : population) {
        for (auto& gene : individual) {
            gene = dis(gen);
        }
    }
}

template <typename Random>
void tlbo(vector<vector<double>>& population, vector<double>& fitnesses, int dim, double lower_bound, double upper_bound, Random& gen, int max_evals) {
    int evals = 0;
    vector<double> best_solution(dim);
    double best_fitness = numeric_limits<double>::max();

    // Find the initial best solution
    for (int i = 0; i < population.size(); i++) {
        double fit = calculateFitness(population[i]);
        fitnesses[i] = fit;
        evals++;

        if (fit < best_fitness) {
            best_fitness = fit;
            best_solution = population[i];
        }
    }

    // Main loop
    while (evals < max_evals) {
        teachingPhase(population, fitnesses, best_solution, best_fitness, dim, lower_bound, upper_bound, gen, evals, max_evals);
        if (evals >= max_evals) break;
        learningPhase(population, fitnesses, dim, lower_bound, upper_bound, gen, evals, max_evals);
    }
}

template <typename Random>
void teachingPhase(vector<vector<double>>& population, vector<double>& fitnesses, vector<double>& best_solution, double& best_fitness, int dim, double lower_bound, double upper_bound, Random& gen, int& evals, int max_evals) {
    uniform_real_distribution<> dis(0, 1);
    vector<double> mean(dim, 0.0);

    // Calculate the mean of the population solutions
    for (const auto& s : population) {
        for (int j = 0; j < dim; j++) {
            mean[j] += s[j];
        }
    }
    for (double &m : mean) m /= population.size();

    // Modify the population solutions
    for (int i = 0; i < population.size(); i++) {
        vector<double> new_solution = population[i];

        // Modify the solution by teaching it
        for (int j = 0; j < dim; j++) {
            double tf = 1 + dis(gen);
            new_solution[j] += dis(gen) * (best_solution[j] - tf * mean[j]);
            clip(new_solution, lower_bound, upper_bound);
        }

        double new_fitness = calculateFitness(new_solution);
        evals++;

        // Update the population if the new solution is better
        if (new_fitness < fitnesses[i]) {
            population[i] = new_solution;
            fitnesses[i] = new_fitness;

            if (new_fitness < best_fitness) {
                best_fitness = new_fitness;
                best_solution = new_solution;
            }
        }
        
        if (evals >= max_evals) return;
    }
}

template <typename Random>
void learningPhase(vector<vector<double>>& population, vector<double>& fitnesses, int dim, double lower_bound, double upper_bound, Random& gen, int& evals, int max_evals) {
    uniform_real_distribution<> dis(0, 1);
    uniform_int_distribution<> index_dist(0, population.size() - 1);

    // Modify the population solutions by learning from another solution
    for (int i = 0; i < population.size(); i++) {
        int j;

        // Select a random solution different from the current one
        do {
            j = index_dist(gen);
        } while (j == i);

        vector<double> new_solution = population[i];
        bool has_changed = false;

        // Modify the solution by changing some of its genes
        for (int k = 0; k < dim; k++) {
            if (dis(gen) < 0.5) {
                double change = dis(gen) * (population[j][k] - population[i][k]);
                new_solution[k] += change;
                clip(new_solution, lower_bound, upper_bound);
                has_changed = true;
            }
        }

        // Update the population if the new solution is better
        if (has_changed) {
            double new_fitness = calculateFitness(new_solution);
            evals++;
            if (new_fitness < fitnesses[i]) {
                population[i] = new_solution;
                fitnesses[i] = new_fitness;
            }
            if (evals >= max_evals) break;
        }
    }
}

double calculateFitness(const vector<double>& solution) {
    return cec17_fitness(const_cast<double*>(solution.data()));
}
