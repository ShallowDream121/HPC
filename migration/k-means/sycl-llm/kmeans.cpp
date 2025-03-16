#include "kmeans.h"

using namespace sycl;

const int MAX_CHAR_PER_LINE = 1024;

class KMEANS {
private:
    int numClusters;
    int numCoords;
    int numObjs;
    int *membership; // [numObjs]
    char *filename;
    float **objects; // [numObjs][numCoords] data objects
    float **clusters; // [numClusters][numCoords] cluster centers
    float threshold;
    int loop_iterations;

public:
    KMEANS(int k);
    void file_read(char *fn);
    void file_write();
    void sycl_kmeans();
    inline int nextPowerOfTwo(int n);
    void free_memory();
    virtual ~KMEANS();
};

KMEANS::~KMEANS() {
    free(membership);
    free(clusters[0]);
    free(clusters);
    free(objects[0]);
    free(objects);
}

KMEANS::KMEANS(int k) {
    threshold = 0.001;
    numObjs = 0;
    numCoords = 0;
    numClusters = k;
    filename = NULL;
    loop_iterations = 0;
}

void KMEANS::file_write() {
    std::ofstream fptr;
    char outFileName[1024];

    // Output: the coordinates of the cluster centers
    sprintf(outFileName, "%s.cluster_centres", filename);
    std::cout << "Writing coordinates of K=" << numClusters << " cluster centers to file \"" << outFileName << "\"\n";
    fptr.open(outFileName);
    for (int i = 0; i < numClusters; i++) {
        fptr << i << " ";
        for (int j = 0; j < numCoords; j++)
            fptr << clusters[i][j] << " ";
        fptr << "\n";
    }
    fptr.close();

    // Output: the closest cluster center to each of the data points
    sprintf(outFileName, "%s.membership", filename);
    std::cout << "Writing membership of N=" << numObjs << " data objects to file \"" << outFileName << "\"\n";
    fptr.open(outFileName);
    for (int i = 0; i < numObjs; i++) {
        fptr << i << " " << membership[i] << "\n";
    }
    fptr.close();
}

inline int KMEANS::nextPowerOfTwo(int n) {
    n--;
    n = n >> 1 | n;
    n = n >> 2 | n;
    n = n >> 4 | n;
    n = n >> 8 | n;
    n = n >> 16 | n;
    return ++n;
}

void KMEANS::sycl_kmeans() {
    queue q;
    int index, loop = 0;
    int *newClusterSize = new int[numClusters]; // [numClusters]: no. objects assigned in each new cluster
    float delta; // % of objects changes their clusters
    float **dimObjects; // [numCoords][numObjs]
    float **dimClusters;
    float **newClusters; // [numCoords][numClusters]

    // Copy objects given in [numObjs][numCoords] layout to new [numCoords][numObjs] layout
    malloc2D(dimObjects, numCoords, numObjs, float);
    for (int i = 0; i < numCoords; i++) {
        for (int j = 0; j < numObjs; j++) {
            dimObjects[i][j] = objects[j][i];
        }
    }

    // Pick first numClusters elements of objects[] as initial cluster centers
    malloc2D(dimClusters, numCoords, numClusters, float);
    for (int i = 0; i < numCoords; i++) {
        for (int j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }

    malloc2D(newClusters, numCoords, numClusters, float);
    memset(newClusters[0], 0, numCoords * numClusters * sizeof(float));

    // SYCL buffers
    buffer<float, 2> bufObjects(dimObjects[0], range<2>(numCoords, numObjs));
    buffer<float, 2> bufClusters(dimClusters[0], range<2>(numCoords, numClusters));
    buffer<int, 1> bufMembership(membership, range<1>(numObjs));
    buffer<int, 1> bufNewClusterSize(newClusterSize, range<1>(numClusters));
    buffer<float, 2> bufNewClusters(newClusters[0], range<2>(numCoords, numClusters));

    do {
        q.submit([&](handler &h) {
            auto accObjects = bufObjects.get_access<access::mode::read>(h);
            auto accClusters = bufClusters.get_access<access::mode::read>(h);
            auto accMembership = bufMembership.get_access<access::mode::read_write>(h);
            auto accNewClusterSize = bufNewClusterSize.get_access<access::mode::read_write>(h);
            auto accNewClusters = bufNewClusters.get_access<access::mode::read_write>(h);

            h.parallel_for(range<1>(numObjs), [=](id<1> i) {
                float min_dist = INFINITY;
                int index = 0;

                // Find the nearest cluster
                for (int j = 0; j < numClusters; j++) {
                    float dist = 0.0f;
                    for (int k = 0; k < numCoords; k++) {
                        float diff = accObjects[k][i] - accClusters[k][j];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        index = j;
                    }
                }

                // Update membership and new clusters
                if (accMembership[i] != index) {
                    accMembership[i] = index;
                    for (int k = 0; k < numCoords; k++) {
                        accNewClusters[k][index] += accObjects[k][i];
                    }
                    accNewClusterSize[index]++;
                }
            });
        });

        q.wait();

        // Update cluster centers
        q.submit([&](handler &h) {
            auto accClusters = bufClusters.get_access<access::mode::write>(h);
            auto accNewClusters = bufNewClusters.get_access<access::mode::read>(h);
            auto accNewClusterSize = bufNewClusterSize.get_access<access::mode::read>(h);

            h.parallel_for(range<2>(numCoords, numClusters), [=](id<2> idx) {
                int k = idx[0];
                int j = idx[1];
                if (accNewClusterSize[j] > 0) {
                    accClusters[k][j] = accNewClusters[k][j] / accNewClusterSize[j];
                }
            });
        });

        q.wait();

        // Reset new clusters and sizes
        memset(newClusters[0], 0, numCoords * numClusters * sizeof(float));
        memset(newClusterSize, 0, numClusters * sizeof(int));

        delta = 0.0f; // Placeholder for delta calculation
    } while (delta > threshold && loop++ < 500);

    loop_iterations = loop + 1;

    // Copy final clusters back to host
    malloc2D(clusters, numClusters, numCoords, float);
    auto hostClusters = bufClusters.get_access<access::mode::read>();
    for (int i = 0; i < numClusters; i++) {
        for (int j = 0; j < numCoords; j++) {
            clusters[i][j] = hostClusters[j][i];
        }
    }

    delete[] newClusterSize;
    free(dimObjects[0]);
    free(dimObjects);
    free(dimClusters[0]);
    free(dimClusters);
    free(newClusters[0]);
    free(newClusters);
}

void KMEANS::file_read(char *fn) {
    std::ifstream infile(fn);
    char line[MAX_CHAR_PER_LINE];
    filename = fn;

    // Count number of objects and coordinates
    while (infile.getline(line, MAX_CHAR_PER_LINE)) {
        numObjs++;
    }
    infile.clear();
    infile.seekg(0);

    // Count number of coordinates
    while (infile.getline(line, MAX_CHAR_PER_LINE)) {
        char *token = strtok(line, " \t\n");
        while (token != nullptr) {
            numCoords++;
            token = strtok(nullptr, " \t\n");
        }
        break;
    }
    infile.clear();
    infile.seekg(0);

    // Allocate memory and read data
    malloc2D(objects, numObjs, numCoords, float);
    int i = 0;
    while (infile.getline(line, MAX_CHAR_PER_LINE)) {
        char *token = strtok(line, " \t\n");
        if (token == nullptr) continue;
        for (int j = 0; j < numCoords; j++) {
            objects[i][j] = atof(token);
            token = strtok(nullptr, " \t\n");
        }
        i++;
    }

    // Initialize membership
    membership = new int[numObjs];
    assert(membership != nullptr);
    memset(membership, -1, numObjs * sizeof(int));
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <numClusters> <inputFile>\n";
        return 1;
    }

    KMEANS kmeans(atoi(argv[1]));
    kmeans.file_read(argv[2]);
    kmeans.sycl_kmeans();
    kmeans.file_write();
    return 0;
}