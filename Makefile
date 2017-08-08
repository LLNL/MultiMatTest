GCC_FLAGS = -O3 -std=c99
NVCC_FLAGS = -O3 -arch=sm_35 -Xcompiler "-O3" -ccbin g++

ifeq ($(DEBUG), yes)
  GCC_FLAGS = -O0 -g -std=c99
  NVCC_FLAGS = -O0 -g -G -arch=sm_35 -Xcompiler "-O0 -g" -ccbin g++
endif

CSRC  = $(wildcard *.c)
CXXSRC  = $(wildcard *.cu)
OBJS = $(patsubst %.c, %.o, $(CSRC))
OBJS += $(patsubst %.cu, %.o, $(CXXSRC))

mm: $(OBJS)
	nvcc $(NVCC_FLAGS) $(OBJS) -o multi

%.o: %.c Makefile
	gcc $(GCC_FLAGS) -c $< -o $@

%.o: %.cu Makefile
	nvcc $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf *.o multi
