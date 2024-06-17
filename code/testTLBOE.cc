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
 * Mutates an individual by adding a random value to a random dimension.
 *
 * @param individual The individual to be mutated.
 * @param num_features The number of dimensions in the individual.
 * @param mutation_rate The rate of mutation.
 * @param lower_bound The lower bound for the individual values.
 * @param upper_bound The upper bound for the individual values.
 * @param gen The random number generator.
 */
template <typename Random>
void mutateIndividual(vector<double>& individual, int num_features, double mutation_rate, double lower_bound, double upper_bound, Random& gen);

/**
 * Replaces the worst half of the population with the best half.
 *
 * @param population The population of solutions.
 * @param fitnesses The fitness values of the solutions.
 * @param num_features The number of dimensions in the solutions.
 * @param mutation_rate The rate of mutation.
 * @param lower_bound The lower bound for the solution values.
 * @param upper_bound The upper bound for the solution values.
 * @param evals The number of fitness evaluations.
 * @param max_evals The maximum number of fitness evaluations.
 * @param gen The random number generator.
 */
template <typename Random>
void elitistReplacement(vector<vector<double>>& population, vector<double>& fitnesses, int num_features, double mutation_rate, double lower_bound, double upper_bound, int& evals, const int& max_evals, Random& gen);

/**
 * Replaces duplicate solutions with new random ones.
 *
 * @param population The population of solutions.
 * @param fitnesses The fitness values of the solutions.
 * @param num_features The number of dimensions in the solutions.
 * @param lower_bound The lower bound for the solution values.
 * @param upper_bound The upper bound for the solution values.
 * @param evals The number of fitness evaluations.
 * @param max_evals The maximum number of fitness evaluations.
 * @param mutation_rate The rate of mutation.
 * @param gen The random number generator.
 */
template <typename Random>
void replaceDuplicates(vector<vector<double>>& population, vector<double>& fitnesses, int num_features, double lower_bound, double upper_bound, int& evals, const int& max_evals, const double mutation_rate, Random& gen);

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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <dimension>" << endl;
        return 1;
    }

    int dim = atoi(argv[1]);
    if (dim != 10 && dim != 30 && dim != 50) {
        cerr << "Dimension must be 10, 30, or 50." << endl;
        return 1;
    }

    const int seed = 43;
    const int T = 10;
    const double lower_bound = -100.0, upper_bound = 100.0;
    const int num_learners = 30;
    const int max_evals = 10000 * dim;
    double mean_fitness = 0.0;

    for (int funcid = 1; funcid <= 30; funcid++) {
        for (int t = 0; t < T; t++) {
            mt19937 gen(seed + t);
            vector<vector<double>> population(num_learners, vector<double>(dim));
            vector<double> fitnesses(num_learners);

            cec17_init("tlboe", funcid, dim);

            initializePopulation(population, fitnesses, dim, lower_bound, upper_bound, gen);
            tlbo(population, fitnesses, dim, lower_bound, upper_bound, gen, max_evals);

            double best_fitness = *min_element(fitnesses.begin(), fitnesses.end());

            cout << "Best Random[F" << funcid << "]: " << ", run: " << t << ", best fitness: " << scientific << best_fitness << endl;
            mean_fitness += best_fitness;
        }
        
        mean_fitness /= T;
        cout << "Mean Random[F" << funcid << "]: " << scientific << cec17_error(mean_fitness) << endl;
        mean_fitness = 0.0;
    }

    return 0;
}

template <typename Random>
void initializePopulation(vector<vector<double>>& population, vector<double>& fitnesses, int dim, double lower_bound, double upper_bound, Random& gen) {
    uniform_real_distribution<> dis(lower_bound, upper_bound);
    for (int i = 0; i < population.size(); i++) {
        for (int j = 0; j < dim; j++) {
            population[i][j] = dis(gen);
        }
        fitnesses[i] = calculateFitness(population[i]);
    }
}

template <typename Random>
void tlbo(vector<vector<double>>& population, vector<double>& fitnesses, int dim, double lower_bound, double upper_bound, Random& gen, int max_evals) {
    int evals = population.size();  // Already evaluated the initial population
    vector<double> best_solution(dim);
    double best_fitness = numeric_limits<double>::max();
    uniform_real_distribution<> dis(0, 1);
    uniform_int_distribution<> tf_dis(1, 2);
    uniform_int_distribution<> index_dist(0, population.size() - 1);

    // Find the teacher
    for (int i = 0; i < population.size(); i++) {
        if (fitnesses[i] < best_fitness) {
            best_fitness = fitnesses[i];
            best_solution = population[i];
        }
    }

    // Main loop
    while (evals < max_evals) {

        // TEACHING PHASE
        // Calculate the mean student
        vector<double> mean(dim, 0.0);
        for (const auto& s : population) {
            for (int j = 0; j < dim; j++) {
                mean[j] += s[j];
            }
        }
        for (double &m : mean) {
            m /= population.size();
        }

        for (int i = 0; i < population.size(); i++) {
            vector<double> new_solution = population[i];

            // Calculate modified student
            double tf = tf_dis(gen);
            double eta = dis(gen);
            for (int j = 0; j < dim; j++) {
                new_solution[j] += eta * (best_solution[j] - tf * mean[j]);
            }
            clip(new_solution, lower_bound, upper_bound);

            double new_fitness = calculateFitness(new_solution);
            evals++;

            if (new_fitness < fitnesses[i]) {
                population[i] = new_solution;
                fitnesses[i] = new_fitness;

                if (new_fitness < best_fitness) {
                    best_fitness = new_fitness;
                    best_solution = new_solution;
                }
            }
            if (evals >= max_evals) break;

            // LEARNING PHASE
            // Choose a different student
            int j;
            do {
                j = index_dist(gen);
            } while (j == i);

            // Calculate again eta
            eta = dis(gen);

            // Calculate the new solution
            if (fitnesses[i] < fitnesses[j]) {
                for (int k = 0; k < dim; k++) {
                    new_solution[k] += eta * (population[i][k] - population[j][k]);
                }
            } else {
                for (int k = 0; k < dim; k++) {
                    new_solution[k] += eta * (population[j][k] - population[i][k]);
                }
            }
            clip(new_solution, lower_bound, upper_bound);

            calculateFitness(new_solution);
            evals++;

            if (new_fitness < fitnesses[i]) {
                population[i] = new_solution;
                fitnesses[i] = new_fitness;
            }

            // Elitist replacement with mutation
            elitistReplacement(population, fitnesses, dim, 0.1, lower_bound, upper_bound, evals, max_evals, gen);
            if (evals >= max_evals) break;

            // Find duplicate solutions and generate new random ones
            replaceDuplicates(population, fitnesses, dim, lower_bound, upper_bound, evals, max_evals, 0.1, gen);
            if (evals >= max_evals) break;
        }
    }
}

double calculateFitness(const vector<double>& solution) {
    return cec17_fitness(const_cast<double*>(solution.data()));
}

template <typename Random>
void mutateIndividual(vector<double>& individual, int num_features, double mutation_rate, double lower_bound, double upper_bound, Random& gen) {
    normal_distribution<> dis(0, 1);
    int num_mutations = static_cast<int>(num_features * mutation_rate);
    vector<int> mutation_dimension(num_features);

    // Randomly shuffle the dimensions
    iota(mutation_dimension.begin(), mutation_dimension.end(), 0);
    shuffle(mutation_dimension.begin(), mutation_dimension.end(), gen);

    for (int m = 0; m < num_mutations; m++) {
        individual[mutation_dimension[m]] += dis(gen);
    }
    clip(individual, lower_bound, upper_bound);
}

template <typename Random>
void elitistReplacement(vector<vector<double>>& population, vector<double>& fitnesses, int num_features, double mutation_rate, double lower_bound, double upper_bound, int& evals, const int& max_evals, Random& gen) {
    int half_pop_size = population.size() / 2;
    for (int i = 0; i < half_pop_size; i++) {
        int second_half_index = i + half_pop_size;
        if (fitnesses[second_half_index] < fitnesses[i]) {
            population[i] = population[second_half_index];
            fitnesses[i] = fitnesses[second_half_index];
            mutateIndividual(population[i], num_features, mutation_rate, lower_bound, upper_bound, gen);
            fitnesses[i] = calculateFitness(population[i]);
            evals++;
            if (evals >= max_evals) break;
        }
    }
}

template <typename Random>
void replaceDuplicates(vector<vector<double>>& population, vector<double>& fitnesses, int num_features, double lower_bound, double upper_bound, int& evals, const int& max_evals, const double mutation_rate, Random& gen) {
    for (int i = 0; i < population.size(); i++) {
        for (int j = i + 1; j < population.size(); j++) {
            if (population[i] == population[j]) {
                mutateIndividual(population[j], num_features, mutation_rate, lower_bound, upper_bound, gen);
                fitnesses[j] = calculateFitness(population[j]);
                evals++;
                if (evals >= max_evals) break;
            }
        }
    }
}
