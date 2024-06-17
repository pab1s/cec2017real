extern "C" {
#include "cec17.h"
}
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

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
void initializePopulation(vector<vector<double>>& population,
                          vector<double>& fitnesses, int dim,
                          double lower_bound, double upper_bound, Random& gen);

/**
 * Calculates the fitness value for a given solution.
 *
 * @param solution The solution for which the fitness value is calculated.
 * @return The fitness value of the solution.
 */
double calculateFitness(const vector<double>& solution);

/**
 * Performs the DEBS heuristic for the Teaching-Learning-Based Optimization
 * (TLBO) algorithm.
 *
 * This function updates the solution by applying the DEBS heuristic to the
 * current solution and the new solution. The DEBS heuristic is a simple
 * heuristic to choose a solution based on the constraints of the solution space
 * and fitnesses.
 *
 * @param solution The current solution.
 * @param new_solution The new solution.
 * @param lower_bound The lower bound of the solution space.
 * @param upper_bound The upper bound of the solution space.
 * @param fitness The fitness value of the current solution.
 * @param new_fitness The fitness value of the new solution.
 * @return The updated solution.
 */
vector<double> debsHeuristic(const vector<double>& solution,
                             const vector<double>& new_solution, double fitness,
                             double new_fitness, double lower_bound,
                             double upper_bound);

/**
 * Performs the teaching phase of the Teaching-Learning-Based Optimization
 * (TLBO) algorithm.
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
void teachingPhase(vector<vector<double>>& population,
                   vector<double>& fitnesses, vector<double>& best_solution,
                   double& best_fitness, int dim, double lower_bound,
                   double upper_bound, Random& gen, int& evals, int max_evals);

/**
 * Performs the learning phase of the Teaching-Learning-Based Optimization
 * (TLBO) algorithm.
 *
 * This function updates the population by applying the teaching and learning
 * phases of the TLBO algorithm. The teaching phase involves selecting a teacher
 * and updating the population based on the teacher's knowledge. The learning
 * phase involves selecting a learner and updating the population based on the
 * learner's knowledge.
 *
 * @param population   The population of solutions.
 * @param fitnesses    The fitness values of the solutions in the population.
 * @param best_solution The best solution found so far.
 * @param best_fitness The fitness value of the best solution found so far.
 * @param dim          The dimensionality of the solutions.
 * @param lower_bound  The lower bound of the solution space.
 * @param upper_bound  The upper bound of the solution space.
 * @param gen          The random number generator.
 * @param evals        The current number of evaluations.
 * @param max_evals    The maximum number of evaluations allowed.
 */
template <typename Random>
void learningPhase(vector<vector<double>>& population,
                   vector<double>& fitnesses, vector<double>& best_solution,
                   double& best_fitness, int dim, double lower_bound,
                   double upper_bound, Random& gen, int& evals, int max_evals);

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
void tlbo(vector<vector<double>>& population, vector<double>& fitnesses,
          int dim, double lower_bound, double upper_bound, Random& gen,
          int max_evals);

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

int main(int argc, char* argv[]) {
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
    const int num_learners = 50;
    const int max_evals = 10000 * dim;
    double mean_fitness = 0.0;

    for (int funcid = 1; funcid <= 30; funcid++) {
        for (int t = 0; t < T; t++) {
            mt19937 gen(seed + t);
            vector<vector<double>> population(num_learners,
                                              vector<double>(dim));
            vector<double> fitnesses(num_learners);

            cec17_init("tlbob", funcid, dim);

            initializePopulation(population, fitnesses, dim, lower_bound,
                                 upper_bound, gen);
            tlbo(population, fitnesses, dim, lower_bound, upper_bound, gen,
                 max_evals);

            double best_fitness =
                *min_element(fitnesses.begin(), fitnesses.end());

            cout << "Best Random[F" << funcid << "]: " << ", run: " << t
                 << ", best fitness: " << scientific << best_fitness << endl;
            mean_fitness += best_fitness;
        }

        mean_fitness /= T;
        cout << "Mean Random[F" << funcid << "]: " << scientific
             << cec17_error(mean_fitness) << endl;
        mean_fitness = 0.0;
    }

    return 0;
}

template <typename Random>
void initializePopulation(vector<vector<double>>& population,
                          vector<double>& fitnesses, int dim,
                          double lower_bound, double upper_bound, Random& gen) {
    uniform_real_distribution<> dis(lower_bound, upper_bound);
    for (auto& individual : population) {
        for (auto& gene : individual) {
            gene = dis(gen);
        }
    }
}

vector<double> debsHeuristic(const vector<double>& solution,
                             const vector<double>& new_solution, double fitness,
                             double new_fitness, double lower_bound,
                             double upper_bound) {
    // Check feasaibility of the solutions
    bool is_solution_feasible = all_of(
        solution.begin(), solution.end(),
        [&](double val) { return val >= lower_bound && val <= upper_bound; });
    bool is_new_solution_feasible = all_of(
        new_solution.begin(), new_solution.end(),
        [&](double val) { return val >= lower_bound && val <= upper_bound; });

    // If both solutions are feasible, choose the one with the better fitness
    if (is_solution_feasible && is_new_solution_feasible) {
        return fitness < new_fitness ? solution : new_solution;
    }

    // If only one solution is feasible, choose the feasible solution
    if (is_solution_feasible) {
        return solution;
    } else if (is_new_solution_feasible) {
        return new_solution;
    }

    // If both solutions are infeasible, choose the one with the smaller
    // constraint violation
    double violation = 0.0;
    double new_violation = 0.0;
    for (int i = 0; i < solution.size(); i++) {
        if (solution[i] < lower_bound) {
            violation += lower_bound - solution[i];
        } else if (solution[i] > upper_bound) {
            violation += solution[i] - upper_bound;
        }

        if (new_solution[i] < lower_bound) {
            new_violation += lower_bound - new_solution[i];
        } else if (new_solution[i] > upper_bound) {
            new_violation += new_solution[i] - upper_bound;
        }
    }

    // Choose the solution with the smaller constraint violation
    vector<double> result = violation < new_violation ? solution : new_solution;
    clip(result, lower_bound, upper_bound);
    return result;
}

template <typename Random>
void tlbo(vector<vector<double>>& population, vector<double>& fitnesses,
          int dim, double lower_bound, double upper_bound, Random& gen,
          int max_evals) {
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
        teachingPhase(population, fitnesses, best_solution, best_fitness, dim,
                      lower_bound, upper_bound, gen, evals, max_evals);
        if (evals >= max_evals) break;
        learningPhase(population, fitnesses, best_solution, best_fitness, dim,
                      lower_bound, upper_bound, gen, evals, max_evals);
    }
}

template <typename Random>
void teachingPhase(vector<vector<double>>& population,
                   vector<double>& fitnesses, vector<double>& best_solution,
                   double& best_fitness, int dim, double lower_bound,
                   double upper_bound, Random& gen, int& evals, int max_evals) {
    uniform_real_distribution<double> dis(0, 1);
    uniform_int_distribution<int> int_dist(1, 2);
    int nm = 20;  // Large nm to keep x_ref close to teacher
    vector<double> mean(dim, 0.0);
    int tf = int_dist(gen);

    // Calculate the mean of the population solutions
    for (const auto& s : population) {
        for (int j = 0; j < dim; j++) {
            mean[j] += s[j];
        }
    }
    for (double& m : mean) m /= population.size();

    // Modify the population solutions
    for (int i = 0; i < population.size(); i++) {
        vector<double> new_solution = population[i];
        double eta = dis(gen);

        // Compute x_ref for each individual
        vector<double> x_ref(dim);
        double r1 = dis(gen);
        double r2 = dis(gen);

        for (int j = 0; j < dim; j++) {
            if ((best_solution[j] - new_solution[j] < 0) ||
                (best_solution[j] - new_solution[j] == 0 && r2 < 0.5))
                x_ref[j] =
                    best_solution[j] + (pow(r1, 1.0 / (nm + 1)) - 1) *
                                           (best_solution[j] - lower_bound);
            else
                x_ref[j] =
                    best_solution[j] + (1 - pow(r1, 1.0 / (nm + 1))) *
                                           (upper_bound - best_solution[j]);
        }

        // Modify the solution by teaching it using the modified reference frame
        for (int j = 0; j < dim; j++) {
            new_solution[j] +=
                eta * (best_solution[j] - tf * mean[j] + (tf - 1) * x_ref[j]);
        }
        clip(new_solution, lower_bound, upper_bound);
        double new_fitness = calculateFitness(new_solution);
        evals++;

        // Apply the Deb's heuristic to update the solution
        vector<double> candidate =
            debsHeuristic(population[i], new_solution, fitnesses[i],
                          new_fitness, lower_bound, upper_bound);

        if (candidate != population[i]) {
            population[i] = candidate;
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
void learningPhase(vector<vector<double>>& population,
                   vector<double>& fitnesses, vector<double>& best_solution,
                   double& best_fitness, int dim, double lower_bound,
                   double upper_bound, Random& gen, int& evals, int max_evals) {
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

        // Modify the solution by changing some of its genes
        double eta = dis(gen);
        for (int k = 0; k < dim; k++) {
            new_solution[k] += eta * (population[j][k] - population[i][k]);
        }

        double new_fitness = calculateFitness(new_solution);
        evals++;

        // Apply the Deb's heuristic to update the solution
        vector<double> candidate =
            debsHeuristic(population[i], new_solution, fitnesses[i],
                          new_fitness, lower_bound, upper_bound);

        if (candidate != population[i]) {
            population[i] = candidate;
            fitnesses[i] = new_fitness;

            if (new_fitness < best_fitness) {
                best_fitness = new_fitness;
                best_solution = new_solution;
            }
        }

        if (evals >= max_evals) break;
    }
}

double calculateFitness(const vector<double>& solution) {
    return cec17_fitness(const_cast<double*>(solution.data()));
}