#pragma once

template <typename Container>
class ResizableDataBuffer {

public:
    explicit ResizableDataBuffer(Container data) : _data(data){}

    using value_type = std::ranges::range_value_t<Container>;

    auto begin() noexcept {
        return _data.begin();
    }

    auto end() noexcept {
        return _data.end();
    }

    value_type* data() noexcept {
        return _data.data();
    }

    [[nodiscard]] size_t size() const noexcept {
        return _data.size();
    }

    void resize(size_t count) {
        _data.get_data().resize(count);
    }

    auto get_data() {
        return _data.get_data();
    }

    Container& underlying() {
        return _data;
    }


private:
    Container _data;
};

