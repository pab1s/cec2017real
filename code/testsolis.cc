extern "C" {
#include "cec17.h"
}
#include <iostream>
#include <random>
#include <vector>

using namespace std;

void clip(vector<double> &sol, int lower, int upper) {
    for (auto &val : sol) {
        if (val < lower) {
            val = lower;
        } else if (val > upper) {
            val = upper;
        }
    }
}

void increm_bias(vector<double> &bias, vector<double> dif) {
    for (unsigned i = 0; i < bias.size(); i++) {
        bias[i] = 0.2 * bias[i] + 0.4 * (dif[i] + bias[i]);
    }
}

void decrement_bias(vector<double> &bias, vector<double> dif) {
    for (unsigned i = 0; i < bias.size(); i++) {
        bias[i] = bias[i] - 0.4 * (dif[i] + bias[i]);
    }
}

/**
 * Aplica el Solis Wets
 *
 * @param  sol solucion a mejorar.
 * @param fitness fitness de la solución.
 */
template <class Random>
void soliswets(vector<double> &sol, double &fitness, double delta, int maxevals,
               int lower, int upper, Random &random) {
    const size_t dim = sol.size();
    vector<double> bias(dim), dif(dim), newsol(dim);
    double newfit;
    size_t i;

    int evals = 0;
    int num_success = 0;
    int num_failed = 0;

    while (evals < maxevals) {
        std::uniform_real_distribution<double> distribution(0.0, delta);

        for (i = 0; i < dim; i++) {
            dif[i] = distribution(random);
            newsol[i] = sol[i] + dif[i] + bias[i];
        }

        clip(newsol, lower, upper);
        newfit = cec17_fitness(&newsol[0]);
        evals += 1;

        if (newfit < fitness) {
            sol = newsol;
            fitness = newfit;
            increm_bias(bias, dif);
            num_success += 1;
            num_failed = 0;
        } else if (evals < maxevals) {
            for (i = 0; i < dim; i++) {
                newsol[i] = sol[i] - dif[i] - bias[i];
            }

            clip(newsol, lower, upper);
            newfit = cec17_fitness(&newsol[0]);
            evals += 1;

            if (newfit < fitness) {
                sol = newsol;
                fitness = newfit;
                decrement_bias(bias, dif);
                num_success += 1;
                num_failed = 0;
            } else {
                for (i = 0; i < dim; i++) {
                    bias[i] /= 2;
                }

                num_success = 0;
                num_failed += 1;
            }
        }

        if (num_success >= 5) {
            num_success = 0;
            delta *= 2;
        } else if (num_failed >= 3) {
            num_failed = 0;
            delta /= 2;
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

    const int T = 10;
    vector<double> sol;
    int seed = 43;
    double mean_fitness = 0;
    int max_evals = 10000 * dim;
    std::uniform_real_distribution<> dis(-100.0, 100.0);

    for (int funcid = 1; funcid <= 30; funcid++) {
        for (int t = 0; t < T; t++) {
            vector<double> sol(dim);
            vector<double> bestsol(dim);
            const size_t maxtimes = 5;
            double fitness, bestfitness = -1;

            cec17_init("solis", funcid, dim);

            // cerr << "Warning: output by console, if you want to create the "
            //        "output file you have to comment cec17_print_output()"
            //     << endl;
            // cec17_print_output(); // Comment to generate the output file

            std::mt19937 gen(seed + t);  // Inicio semilla

            for (size_t times = 0; times < maxtimes; times++) {
                for (int i = 0; i < dim; i++) {
                    sol[i] = dis(gen);
                }

                fitness = cec17_fitness(&sol[0]);
                soliswets(sol, fitness, 0.2, max_evals / maxtimes - 1, -100,
                          100, gen);

                if (bestfitness < 0 || fitness < bestfitness) {
                    bestfitness = fitness;
                }
            }
            cout << "Best Fitness[F" << funcid << "]: " << ", run: " << t + 1
                 << ", fitness: " << scientific << cec17_error(bestfitness)
                 << endl;
            mean_fitness += bestfitness;
        }
        mean_fitness /= T;
        cout << "Mean Fitness[F" << funcid << "]: " << scientific
             << cec17_error(mean_fitness) << endl;
        mean_fitness = 0;
    }

    return 0;
}
