#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>

extern "C" {
#include "cec17.h"
}

using namespace std;

void initializePopulation(vector<vector<double>>& population, vector<double>& fitnesses, int dim, double lower_bound, double upper_bound, mt19937& gen);
double calculateFitness(const vector<double>& solution);
void teachingPhase(vector<vector<double>>& population, vector<double>& fitnesses, vector<double>& best_solution, double& best_fitness, int dim, double lower_bound, double upper_bound, mt19937& gen, int& evals, int max_evals);
void learningPhase(vector<vector<double>>& population, vector<double>& fitnesses, int dim, double lower_bound, double upper_bound, mt19937& gen, int& evals, int max_evals);

int main() {
    const int dim = 10;  // Problem dimensionality
    const int num_learners = 30;  // Population size
    const double lower_bound = -100.0, upper_bound = 100.0;  // Search space bounds
    int max_evals = 10000 * dim;
    int evals = 0;

    random_device rd;
    mt19937 gen(rd());

    vector<vector<double>> population(num_learners, vector<double>(dim));
    vector<double> fitnesses(num_learners);
    vector<double> best_solution(dim);
    double best_fitness = numeric_limits<double>::max();

    cec17_init("test", 1, dim);

    initializePopulation(population, fitnesses, dim, lower_bound, upper_bound, gen);
    for (int i = 0; i < num_learners; i++) {
        fitnesses[i] = calculateFitness(population[i]);
        if (fitnesses[i] < best_fitness) {
            best_fitness = fitnesses[i];
            best_solution = population[i];
        }
    }
    evals += num_learners;

    while (evals < max_evals) {
        teachingPhase(population, fitnesses, best_solution, best_fitness, dim, lower_bound, upper_bound, gen, evals, max_evals);
        if (evals >= max_evals) break;
        learningPhase(population, fitnesses, dim, lower_bound, upper_bound, gen, evals, max_evals);
    }

    cout << "Best Solution: ";
    for (const auto& gene : best_solution) cout << gene << " ";
    cout << "\nBest Fitness: " << best_fitness << endl;
    return 0;
}

void initializePopulation(vector<vector<double>>& population, vector<double>& fitnesses, int dim, double lower_bound, double upper_bound, mt19937& gen) {
    uniform_real_distribution<> dis(lower_bound, upper_bound);
    for (auto& individual : population) {
        for (auto& gene : individual) {
            gene = dis(gen);
        }
    }
}

double calculateFitness(const vector<double>& solution) {
    return cec17_fitness(const_cast<double*>(solution.data()));
}

void teachingPhase(vector<vector<double>>& population, vector<double>& fitnesses, vector<double>& best_solution, double& best_fitness, int dim, double lower_bound, double upper_bound, mt19937& gen, int& evals, int max_evals) {
    uniform_real_distribution<> dis(0, 1);
    vector<double> mean(dim, 0.0);
    for (const auto& s : population) {
        for (int j = 0; j < dim; j++) {
            mean[j] += s[j];
        }
    }
    for (double &m : mean) m /= population.size();

    for (int i = 0; i < population.size(); i++) {
        vector<double> new_solution = population[i];
        for (int j = 0; j < dim; j++) {
            double tf = 1 + dis(gen);
            new_solution[j] += dis(gen) * (best_solution[j] - tf * mean[j]);
            new_solution[j] = clamp(new_solution[j], lower_bound, upper_bound);
        }
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
        if (evals >= max_evals) return;
    }
}

void learningPhase(vector<vector<double>>& population, vector<double>& fitnesses, int dim, double lower_bound, double upper_bound, mt19937& gen, int& evals, int max_evals) {
    uniform_real_distribution<> dis(0, 1);
    uniform_int_distribution<> index_dist(0, population.size() - 1);
    for (int i = 0; i < population.size(); i++) {
        int j;
        do {
            j = index_dist(gen);
        } while (j == i);

        vector<double> new_solution = population[i];
        bool has_changed = false;

        for (int k = 0; k < dim; k++) {
            if (dis(gen) < 0.5) {
                double change = dis(gen) * (population[j][k] - population[i][k]);
                new_solution[k] += change;
                new_solution[k] = clamp(new_solution[k], lower_bound, upper_bound);
                has_changed = true;
            }
        }

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
