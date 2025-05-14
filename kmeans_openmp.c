
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <omp.h>

static inline double distance(const double* point, const double* centroid, int n_features) {
    double dist = 0.0;
    for (int f = 0; f < n_features; f++) {
        double diff = point[f] - centroid[f];
        dist += diff * diff;
    }
    return dist;
}

void initialize_kmeans_plus_plus(
    const double* data,
    int n_samples,
    int n_features,
    int k,
    double* centroids,
    int random_state
) {
    srand(random_state);
    
    int first_idx = rand() % n_samples;
    for (int f = 0; f < n_features; f++) {
        centroids[f] = data[first_idx * n_features + f];
    }

    double* min_distances = (double*)malloc(n_samples * sizeof(double));
    double* distances = (double*)malloc(n_samples * sizeof(double));
    
    // Initialize min_distances to infinity
    for (int i = 0; i < n_samples; i++) {
        min_distances[i] = DBL_MAX;
    }
    
    for (int c = 1; c < k; c++) {
        #pragma omp parallel for
        for (int i = 0; i < n_samples; i++) {
            distances[i] = distance(&data[i * n_features], &centroids[(c-1) * n_features], n_features);
            if (distances[i] < min_distances[i]) {
                min_distances[i] = distances[i];
            }
        }
        
        double sum_distances = 0.0;
        for (int i = 0; i < n_samples; i++) {
            sum_distances += min_distances[i];
        }
        
        double threshold = ((double)rand() / RAND_MAX) * sum_distances;
        sum_distances = 0.0;
        int next_idx = 0;
        
        for (int i = 0; i < n_samples; i++) {
            sum_distances += min_distances[i];
            if (sum_distances >= threshold) {
                next_idx = i;
                break;
            }
        }
        
        for (int f = 0; f < n_features; f++) {
            centroids[c * n_features + f] = data[next_idx * n_features + f];
        }
    }
    
    free(min_distances);
    free(distances);
}

/**
 * 
 * @param data 
 * @param n_samples 
 * @param n_features 
 * @param k 
 * @param max_iter 
 * @param tol 
 * @param labels 
 * @param centroids 
 * @param random_state 
 */
void kmeans_omp(
    double* data,
    int n_samples,
    int n_features,
    int k,
    int max_iter,
    double tol,
    int* labels,
    double* centroids,
    int random_state
) {

    if (data == NULL || labels == NULL || centroids == NULL || 
        n_samples <= 0 || n_features <= 0 || k <= 0 || max_iter <= 0 || tol < 0) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }

    initialize_kmeans_plus_plus(data, n_samples, n_features, k, centroids, random_state);

    double* new_centroids = (double*)calloc(k * n_features, sizeof(double));
    int* counts = (int*)calloc(k, sizeof(int));
    int* old_labels = (int*)malloc(n_samples * sizeof(int));

    for (int i = 0; i < n_samples; i++) {
        labels[i] = -1;
    }
    
#ifdef DEBUG_KMEANS
    printf("K-means starting with %d samples, %d features, %d clusters\n", n_samples, n_features, k);
    printf("Using %d OpenMP threads\n", omp_get_max_threads());
#endif
    
    for (int iter = 0; iter < max_iter; iter++) {
        memcpy(old_labels, labels, n_samples * sizeof(int));
        
        memset(new_centroids, 0, k * n_features * sizeof(double));
        memset(counts, 0, k * sizeof(int));
        
        #pragma omp parallel
        {

            double* local_centroids = (double*)calloc(k * n_features, sizeof(double));
            int* local_counts = (int*)calloc(k, sizeof(int));
            
            #pragma omp for schedule(static)
            for (int i = 0; i < n_samples; i++) {
                double min_dist = DBL_MAX;
                int best_cluster = 0;

                for (int c = 0; c < k; c++) {
                    double dist = distance(&data[i * n_features], &centroids[c * n_features], n_features);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = c;
                    }
                }
                labels[i] = best_cluster;
                
                local_counts[best_cluster]++;
                for (int f = 0; f < n_features; f++) {
                    local_centroids[best_cluster * n_features + f] += data[i * n_features + f];
                }
            }
            

            #pragma omp critical
            {
                for (int c = 0; c < k; c++) {
                    counts[c] += local_counts[c];
                    for (int f = 0; f < n_features; f++) {
                        new_centroids[c * n_features + f] += local_centroids[c * n_features + f];
                    }
                }
            }
            

            free(local_centroids);
            free(local_counts);
        }
        

        double total_shift = 0.0;
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {

                for (int f = 0; f < n_features; f++) {
                    double old_val = centroids[c * n_features + f];
                    double new_val = new_centroids[c * n_features + f] / counts[c];
                    centroids[c * n_features + f] = new_val;
                    total_shift += (old_val - new_val) * (old_val - new_val);
                }
            } else {

                #ifdef DEBUG_KMEANS
                printf("Warning: Empty cluster %d found in iteration %d\n", c, iter);
                #endif
                
                int random_idx = rand() % n_samples;
                for (int f = 0; f < n_features; f++) {
                    centroids[c * n_features + f] = data[random_idx * n_features + f];
                }
            }
        }
        

        total_shift = sqrt(total_shift);
        
        int label_changes = 0;
        for (int i = 0; i < n_samples; i++) {
            if (labels[i] != old_labels[i]) {
                label_changes++;
            }
        }
        
#ifdef DEBUG_KMEANS
        printf("Iteration %d: %d label changes, shift = %.8f\n", 
               iter, label_changes, total_shift);
#endif
        
        if (total_shift < tol || label_changes == 0) {
#ifdef DEBUG_KMEANS
            printf("Converged after %d iterations\n", iter + 1);
#endif
            break;
        }
    }
    
    // Free memory
    free(new_centroids);
    free(counts);
    free(old_labels);
} 