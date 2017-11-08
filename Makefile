COMPILER = icpc
GCC_FLAGS = -O3 -fopenmp -std=c99
GXX_FLAGS = -O3 -mcpu=power8 -mtune=power8 -fopenmp -std=c++11 
XLC_FLAGS = -O5 -qsmp=omp -qarch=pwr8 -qtune=pwr8 -std=c99
XLX_FLAGS = -O5 -qsmp=omp -qarch=pwr8 -qtune=pwr8 #-std=c++11
ICC_FLAGS = -O3 -qopenmp -std=c99 -xMIC-AVX512
IXX_FLAGS = -O3 -qopenmp -std=c++11 -xhost -qopt-report=5

ifeq ($(DEBUG), yes)
  GCC_FLAGS = -O0 -g -fopenmp -std=c99
  GXX_FLAGS = -O0 -g -fopenmp -std=c++11
  XLC_FLAGS = -O0 -g -qsmp=omp -qarch=pwr8 -qtune=pwr8 -std=c99
  XLX_FLAGS = -O0 -g -qsmp=omp -qarch=pwr8 -qtune=pwr8 -std=c++11
  ICC_FLAGS = -O0 -g -qopenmp -std=c99 -xMIC-AVX512
  IXX_FLAGS = -O0 -g -qopenmp -std=c++11 -xMIC-AVX512
endif

FLAGS = $(IXX_FLAGS)

CSRC  = $(wildcard *.c)
CXXSRC  = $(wildcard *.cpp)
OBJS = $(patsubst %.c, %.o, $(CSRC))
OBJS += $(patsubst %.cpp, %.o, $(CXXSRC))

mm: $(OBJS)
	$(COMPILER) $(FLAGS) $(OBJS) -o multi

%.o: %.c Makefile
	$(COMPILER) $(FLAGS) -c $< -o $@

%.o: %.cpp Makefile
	$(COMPILER) $(FLAGS) -c $< -o $@

clean:
	rm -rf *.o multi
