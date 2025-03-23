# Makefile for CUDA version of KMeans

# Compiler and flags
NVCC        = nvcc
DFLAGS      =
OPTFLAGS    = -O2  
INCFLAGS    = -I.
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=1
NVCCFLAGS 	= $(CFLAGS) --ptxas-options=-v -arch=sm_86
LDFLAGS     = $(OPTFLAGS)
LIBS        =

# CUDA source files
CUDA_C_SRC  = cuda_main.cu cuda_io.cu cuda_wtime.cu
CUDA_CU_SRC = cuda_kmeans.cu

# Object files
CUDA_C_OBJ  = $(CUDA_C_SRC:%.cu=%.o)
CUDA_CU_OBJ = $(CUDA_CU_SRC:%.cu=%.o)

# Default target
target: cuda_main

# Rule to compile CUDA source files
%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

# Link object files to create the executable
cuda_main: $(CUDA_C_OBJ) $(CUDA_CU_OBJ)
	$(NVCC) $(LDFLAGS) -o $@ $(CUDA_C_OBJ) $(CUDA_CU_OBJ) $(LIBS)

# Test target: run cuda_main with ../Image_data/color100.txt
test: cuda_main
	./cuda_main -i ../Image_data/color100.txt -n 4 -t 0.001

# Clean up build artifacts
clean:
	rm -rf *.o cuda_main \
	       core* .make.state gmon.out \
               *.cluster_centres *.membership \
               Image_data/*.cluster_centres \
               Image_data/*.membership \
               profiles/