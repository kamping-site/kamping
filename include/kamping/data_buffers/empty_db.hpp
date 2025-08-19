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

    [[nodiscard]] size_t size() const {
        return _data.size();
    }

    std::vector<T>& get_data() {
        return _data;
    }


private:
    std::vector<T> _data;
};

