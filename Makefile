CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic -O2
TARGET := fp_growth
SOURCES := main.cpp data_loader.cpp preprocessing.cpp fp_tree.cpp classifier.cpp output.cpp

.PHONY: all clean run-iris

all: $(TARGET)

$(TARGET): $(SOURCES) common.hpp data_loader.hpp preprocessing.hpp fp_tree.hpp classifier.hpp output.hpp
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

run-iris: $(TARGET)
	./$(TARGET) iris/iris.data 0.2 0.7

clean:
	rm -f $(TARGET)
