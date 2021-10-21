#include <iostream>
#include <vector>

#include "wrapper.hpp"

struct S {
  int a = 0;
  int b = 0;
  friend std::ostream& operator<<(std::ostream& out, const S& s) {
    return out << "(" << s.a << ", " << s.b << ")";
  }
};

int main() {
  MPI_Init(nullptr, nullptr);
  MPIWrapper::MPIContext ctx{MPI_COMM_WORLD};
  MPI_Finalize();
}
