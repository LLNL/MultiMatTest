COMPILER = CC
GCC_FLAGS = -O3 -fopenmp -std=c99
GXX_FLAGS = -O3 -fopenmp -std=c++11
ICC_FLAGS = -O3 -qopenmp -std=c99 #-xMIC-AVX512
IXX_FLAGS = -O3 -qopenmp -std=c++11 -qopt-report=5 #-xMIC-AVX512

ifeq ($(DEBUG), yes)
  GCC_FLAGS = -O0 -g -fopenmp -std=c99
  GXX_FLAGS = -O0 -g -fopenmp -std=c++11
  ICC_FLAGS = -O0 -g -qopenmp -std=c99 -xMIC-AVX512
  IXX_FLAGS = -O0 -g -qopenmp -std=c++11 -xMIC-AVX512
endif

CSRC  = $(wildcard *.c)
CXXSRC  = $(wildcard *.cpp)
OBJS = $(patsubst %.c, %.o, $(CSRC))
OBJS += $(patsubst %.cpp, %.o, $(CXXSRC))

mm: $(OBJS)
	$(COMPILER) $(IXX_FLAGS) $(OBJS) -o multi

%.o: %.c Makefile
	$(COMPILER) $(IXX_FLAGS) -c $< -o $@

%.o: %.cpp Makefile
	$(COMPILER) $(IXX_FLAGS) -c $< -o $@

clean:
	rm -rf *.o multi
