#include <vector>


template<typename T>
class EmptyDataBuffer {

public:
    EmptyDataBuffer() : _data(std::vector<T>()){}

    using value_type = T;

    auto begin() {
        return _data.begin();
    }

    auto end() {
        return _data.end();
    }

    T* data() {
        return _data.data();
    }

    size_t size() {
        return _data.size();
    }

    void resize(size_t count) {
        _data.resize(count);
    }


private:
    std::vector<T> _data;
};

