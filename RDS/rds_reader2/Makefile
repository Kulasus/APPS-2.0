SOURCES=$(wildcard *.cpp)
OBJS=$(SOURCES:%.cpp=%.o)
TARGETS=$(SOURCES:%.cpp=%)

CPPFLAGS=-g -pthread -std=c++11 -Wall 
LDFLAGS=-g -pthread -lrt

.PHONY: all clean 

all: $(TARGETS)

%.o: %.cpp
	g++ -c $(CPPFLAGS) $^ -o $@
	
$(TARGETS): %: %.o
	g++  $^ -o $@ $(LDFLAGS)

clean:
	rm -rf *.o $(TARGETS)
