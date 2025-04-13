#include <CL/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "kmeans.h"

// Helper function to compute the next power of two
static inline int nextPowerOfTwo(int n) {
    n--;
    n = n >> 1 | n;
    n = n >> 2 | n;
    n = n >> 4 | n;
    n = n >> 8 | n;
    n = n >> 16 | n;
    return ++n;
}

// Euclidean distance squared between two points
static float euclid_dist_2(int numCoords, int numObjs, int numClusters,
                           const float* objects, const float* clusters,
                           int objectId, int clusterId) {
    float ans = 0.0f;
    for (int i = 0; i < numCoords; i++) {
        float diff = objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId];
        ans += diff * diff;
    }
    return ans;
}

// SYCL kernel to find the nearest cluster for each object
class FindNearestCluster {
public:
    FindNearestCluster(int numCoords, int numObjs, int numClusters,
                       const float* objects, const float* clusters,
                       int* membership, int* intermediates)
        : numCoords_(numCoords), numObjs_(numObjs), numClusters_(numClusters),
          objects_(objects), clusters_(clusters),
          membership_(membership), intermediates_(intermediates) {}

    void operator()(sycl::nd_item<1> item) const {
        int objectId = item.get_global_id(0);
        if (objectId < numObjs_) {
            float min_dist = euclid_dist_2(numCoords_, numObjs_, numClusters_,
                                           objects_, clusters_, objectId, 0);
            int index = 0;

            for (int i = 1; i < numClusters_; i++) {
                float dist = euclid_dist_2(numCoords_, numObjs_, numClusters_,
                                           objects_, clusters_, objectId, i);
                if (dist < min_dist) {
                    min_dist = dist;
                    index = i;
                }
            }

            if (membership_[objectId] != index) {
                intermediates_[item.get_local_id(0)] = 1;
            } else {
                intermediates_[item.get_local_id(0)] = 0;
            }

            membership_[objectId] = index;
        }
    }

private:
    int numCoords_, numObjs_, numClusters_;
    const float* objects_;
    const float* clusters_;
    int* membership_;
    int* intermediates_;
};

// SYCL kernel to compute delta (reduction)
class ComputeDelta {
public:
    ComputeDelta(int* intermediates, int numIntermediates)
        : intermediates_(intermediates), numIntermediates_(numIntermediates) {}

    void operator()(sycl::nd_item<1> item, sycl::local_accessor<int, 1> localData) const {
        int tid = item.get_local_id(0);
        int gid = item.get_global_id(0);

        // Load data into local memory
        if (gid < numIntermediates_) {
            localData[tid] = intermediates_[gid];
        } else {
            localData[tid] = 0;
        }
        item.barrier(sycl::access::fence_space::local_space);

        // Perform reduction
        for (int s = item.get_local_range(0) / 2; s > 0; s >>= 1) {
            if (tid < s) {
                localData[tid] += localData[tid + s];
            }
            item.barrier(sycl::access::fence_space::local_space);
        }

        // Write result back to global memory
        if (tid == 0) {
            intermediates_[0] = localData[0];
        }
    }

private:
    int* intermediates_;
    int numIntermediates_;
};

// Main K-means function
float** sycl_kmeans(float** objects, int numCoords, int numObjs, int numClusters,
    float threshold, int* membership, int* loop_iterations) {
    int i, j, index, loop = 0;
    int* newClusterSize = (int*)calloc(numClusters, sizeof(int));
    float delta;
    float** dimObjects;
    float** clusters;
    float** dimClusters;
    float** newClusters;

    // Allocate memory for 2D arrays
    malloc2D(dimObjects, numCoords, numObjs, float);
    malloc2D(dimClusters, numCoords, numClusters, float);
    malloc2D(newClusters, numCoords, numClusters, float);

    // Initialize data
    for (i = 0; i < numCoords; i++) {
    for (j = 0; j < numObjs; j++) {
    dimObjects[i][j] = objects[j][i];
    }
    }

    for (i = 0; i < numCoords; i++) {
    for (j = 0; j < numClusters; j++) {
    dimClusters[i][j] = dimObjects[i][j];
    }
    }

    for (i = 0; i < numObjs; i++) membership[i] = -1;
    memset(newClusters[0], 0, numCoords * numClusters * sizeof(float));

    // SYCL queue and buffers
    sycl::queue q;
    {
        sycl::buffer<float, 1> deviceObjects(dimObjects[0], sycl::range<1>(numCoords * numObjs));
        sycl::buffer<float, 1> deviceClusters(dimClusters[0], sycl::range<1>(numCoords * numClusters));
        sycl::buffer<int, 1> deviceMembership(membership, sycl::range<1>(numObjs));
        sycl::buffer<int, 1> deviceIntermediates(sycl::range<1>(nextPowerOfTwo(numObjs)));

        const int numThreadsPerClusterBlock = 128;
        const int numClusterBlocks = (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;

        do {
            q.submit([&](sycl::handler& h) {
                auto objects_acc = deviceObjects.get_access<sycl::access::mode::read>(h);
                auto clusters_acc = deviceClusters.get_access<sycl::access::mode::read>(h);
                auto membership_acc = deviceMembership.get_access<sycl::access::mode::read_write>(h);
                auto intermediates_acc = deviceIntermediates.get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::nd_range<1>(numClusterBlocks * numThreadsPerClusterBlock, numThreadsPerClusterBlock),
                            FindNearestCluster(numCoords, numObjs, numClusters,
                                                objects_acc.get_pointer(), clusters_acc.get_pointer(),
                                                membership_acc.get_pointer(), intermediates_acc.get_pointer()));
            });
            
            q.submit([&](sycl::handler& h) {
                auto intermediates_acc = deviceIntermediates.get_access<sycl::access::mode::read_write>(h);
                sycl::local_accessor<int, 1> localData(sycl::range<1>(numThreadsPerClusterBlock), h);
            
                h.parallel_for(sycl::nd_range<1>(numThreadsPerClusterBlock, numThreadsPerClusterBlock),
                               [=](sycl::nd_item<1> item) {
                                   ComputeDelta computeDelta(intermediates_acc.get_pointer(), numClusterBlocks);
                                   computeDelta(item, localData);
                               });
            });
            q.wait(); // while running kernels, segamention fault happened
            // Compute delta
            auto intermediates_host = deviceIntermediates.get_access<sycl::access::mode::read>();
            delta = static_cast<float>(intermediates_host[0]) / numObjs;
        
            // Update membership and cluster centers
            auto membership_host = deviceMembership.get_access<sycl::access::mode::read>();
            for (i = 0; i < numObjs; i++) {
                index = membership_host[i];
                newClusterSize[index]++;
                for (j = 0; j < numCoords; j++) {
                    newClusters[j][index] += objects[i][j];
                }
            }
            for (i = 0; i < numClusters; i++) {
                for (j = 0; j < numCoords; j++) {
                    if (newClusterSize[i] > 0)
                        dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
                    newClusters[j][i] = 0.0f;
                }
                newClusterSize[i] = 0;
            }
        } while (delta > threshold && loop++ < 500);
    }

    *loop_iterations = loop + 1;

    // Allocate and return final clusters
    malloc2D(clusters, numClusters, numCoords, float);
    for (i = 0; i < numClusters; i++) {
    for (j = 0; j < numCoords; j++) {
    clusters[i][j] = dimClusters[j][i];
    }
    }

    // Free memory
    free(dimObjects[0]);
    free(dimObjects);
    free(dimClusters[0]);
    free(dimClusters);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}