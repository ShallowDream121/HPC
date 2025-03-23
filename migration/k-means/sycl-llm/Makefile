# Makefile for SYCL version of KMeans

# Compiler and flags
CXX         = icpx
DFLAGS      =
OPTFLAGS    = -O2
INCFLAGS    = -I.
CXXFLAGS    = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS) -fsycl
LDFLAGS     = $(OPTFLAGS) -fsycl
LDFLAGS += -L/opt/intel/oneapi/compiler/latest/linux/lib -lsycl
LIBS        =

# Source files
SRC         = sycl_main.cpp sycl_io.cpp sycl_wtime.cpp sycl_kmeans.cpp

# Object files
OBJ         = $(SRC:%.cpp=%.o)

# Default target
target: sycl_main

# Rule to compile C++ source files
%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

# Link object files to create the executable
sycl_main: $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ) $(LIBS)

# Test target: run sycl_main with ../Image_data/color100.txt
test: sycl_main
	./sycl_main -i ../Image_data/color100.txt -n 4 -t 0.001

# Clean up build artifacts
clean:
	rm -rf *.o sycl_main \
	       core* .make.state gmon.out \
               *.cluster_centres *.membership \
               Image_data/*.cluster_centres \
               Image_data/*.membership \
               profiles/