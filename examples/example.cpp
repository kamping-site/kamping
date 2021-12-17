#include "kamping/kassert.hpp"

int a() { return 1; }
int b() { return 2;}

int main() {
    KASSERT(a() == b(), "", kamping::assert::normal);
}
