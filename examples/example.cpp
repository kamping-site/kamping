#include "kamping/wrapper.hpp"
#include "kamping/assert.hpp"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>


void printResult(int rank, std::vector<int>& recvData, std::string name) {
    std::stringstream ss;
    ss << rank << ": " << name << ": [";
    for (auto elem: recvData) {
        ss << elem << ", ";
    }
    ss << "]" << std::endl;
    std::cout << ss.str() << std::flush;
}

void printResult(int rank, std::unique_ptr<int[]>& recvData, size_t size, std::string name) {
    std::stringstream ss;
    ss << rank << ": " << name << ": [";
    for (size_t i = 0; i < size; ++i) {
        ss << recvData.get()[i] << ", ";
    }
    ss << "]" << std::endl;
    std::cout << ss.str() << std::flush;
}

using namespace kamping;

bool f() { return true; }
bool g() { return false; }

int main() {
    std::vector<std::pair<int, int>> a{{1, 2}, {2, 3}};
    std::vector<std::pair<int, int>> b{{3, 3}, {4, 5}};

    //KASSERT(a != a || g() == f(), "this work!" << "right?", assert::lightweight);
    KTHROW(a != a, "ok", assert::DefaultException);
}
