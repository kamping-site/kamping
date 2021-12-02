#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>
// TODO Probably rename and reorganize all of this

enum class ptraits { in, out, root, recvCounts, recvDispls };

// trait selector *************************************************************
// returns the index of the first argument type that has the appropriate par_type
template<ptraits trait, size_t I, class Arg, class... Args>
constexpr size_t find_pos() {
    if constexpr(Arg::par_type == trait)
        return I;
    else
        return find_pos<trait, I + 1, Args...>();
}

// returns the first parameter whose type has the appropriate par_type
template<ptraits trait, class... Args>
decltype(auto) select_trait(Args &&... args) {
    return std::move(std::get<find_pos<trait, 0, Args...>()>(std::forward_as_tuple(args...)));
}

template<class T>
struct in_named_tuple {
    T *ptr;
    size_t size;
};

template<class T>
class in_type_ptr {
public:
    // each class contains its type as par_type (must be known at compile time)
    static constexpr ptraits par_type = ptraits::in;
    using value_type = T;

    in_type_ptr(const T *ptr, size_t size) : _ptr(ptr), _size(size) {}

    in_named_tuple<T> get() {
        return {_ptr, _size};
    }

private:
    T *_ptr;
    size_t _size;
};

template<class T>
class in_type_vec {
public:
    // each class contains its type as par_type (must be known at compile time)
    static constexpr ptraits par_type = ptraits::in;
    using value_type = T;

    in_type_vec(std::vector<T> &vec) : _vec(vec) {}

    in_named_tuple<T> get() {
        return {_vec.data(), _vec.size()};
    }

private:
    std::vector<T> &_vec;
};

template<class T>
in_type_vec<T> in(std::vector<T> &vec) {
    return in_type_vec<T>(vec);
}

template<class T>
in_type_ptr<T> in(const T *ptr, size_t size) {
    return in_type_ptr<T>(ptr, size);
}

// different specialized output wrapper
template<class T, ptraits ptrait>
class out_vector {
public:
    static constexpr ptraits par_type = ptrait;
    using value_type = T;
    static constexpr bool isExtractable = false;
    out_vector(std::vector<T> &vec) : vec_ref(vec) {}
    T *get_ptr(size_t s) {
        if(vec_ref.size() < s)
            vec_ref.resize(s);
        return vec_ref.data();
    }

private:
    std::vector<T> &vec_ref;
};

template<class T, ptraits ptrait>
class out_unique {
public:
    static constexpr ptraits par_type = ptrait;
    using value_type = T;
    static constexpr bool isExtractable = false;
    out_unique(std::unique_ptr<T[]> &ptr) : ptr_ref(ptr) {}
    T *get_ptr([[maybe_unused]] size_t s) {
        // it is not so easy to check if the size of a buffer
        // is large enough
        return ptr_ref.get();
    }

private:
    std::unique_ptr<T[]> &ptr_ref;
};

template<class T, ptraits ptrait>
class out_vector_unspecified {
public:
    static constexpr ptraits par_type = ptrait;
    using value_type = T;
    static constexpr bool isExtractable = true;
    out_vector_unspecified() {}
    T *get_ptr(size_t s) {
        vec_ref.resize(s);
        return vec_ref.data();
    }
    std::vector<T> extract() {
        return std::move(vec_ref);
    }
    operator std::vector<T>() {
        return std::move(vec_ref);
    }

private:
    std::vector<T> vec_ref;
};

template<class T, ptraits ptrait>
class out_vector_alternative {
public:
    static constexpr ptraits par_type = ptrait;
    using value_type = T;
    static constexpr bool isExtractable = true;
    out_vector_alternative(std::vector<T> &&rref) : vec_ref(std::forward<std::vector<T>>(rref)) {}
    T *get_ptr(size_t s) {
        vec_ref.resize(s);
        return vec_ref.data();
    }
    std::vector<T> extract() {
        return std::move(vec_ref);
    }
    operator std::vector<T>() {
        return std::move(vec_ref);
    }

private:
    std::vector<T> vec_ref;
};

template<class T, ptraits ptrait>
class out_unique_unspecified {
public:
    static constexpr ptraits par_type = ptrait;
    using value_type = T;
    static constexpr bool isExtractable = true;
    out_unique_unspecified() {}
    T *get_ptr(size_t s) {
        ptr_ref = std::make_unique<T[]>(s);
        return ptr_ref.get();
    }
    std::unique_ptr<T[]> extract() {
        return std::move(ptr_ref);
    }
    operator std::unique_ptr<T[]>() {
        return std::move(ptr_ref);
    }

private:
    std::unique_ptr<T[]> ptr_ref;
};


// something to make this prettier
template<class T>
out_vector<T, ptraits::out> out(std::vector<T> &vec) {
    return out_vector<T, ptraits::out>(vec);
}

template<class T>
out_unique<T, ptraits::out> out(std::unique_ptr<T[]> &ptr) {
    return out_unique<T, ptraits::out>(ptr);
}

template<class T>
class new_vector {
public:
    new_vector() {}
};
template<class T>
out_vector_unspecified<T, ptraits::out> out([[maybe_unused]] new_vector<T> &&) {
    return out_vector_unspecified<T, ptraits::out>();
}

template<class T>
class new_pointer {
public:
    new_pointer() {}
};
template<class T>
out_unique_unspecified<T, ptraits::out> out([[maybe_unused]] new_pointer<T> &&) {
    return out_unique_unspecified<T, ptraits::out>();
}

template<class T>
out_vector_alternative<T, ptraits::out> out(std::vector<T> &&vec) {
    return out_vector_alternative<T, ptraits::out>(std::forward<std::vector<T>>(vec));
}

template<class T>
out_unique_unspecified<T, ptraits::out> out([[maybe_unused]] T *ptr) {
    return out_unique_unspecified<T, ptraits::out>();
}

class in_root {
public:
    static constexpr ptraits par_type = ptraits::root;
    in_root(int root) : _root(root) {}

    int getRoot() {
        return _root;
    }

private:
    int _root;
};

in_root root(int root_) {
    return in_root(root_);
}


template<class T>
out_vector<T, ptraits::recvDispls> recv_displs(std::vector<T> &vec) {
    return out_vector<T, ptraits::recvDispls>(vec);
}

template<class T>
out_unique<T, ptraits::recvDispls> recv_displs(std::unique_ptr<T[]> &ptr) {
    return out_unique<T, ptraits::recvDispls>(ptr);
}

template<class T>
out_vector_unspecified<T, ptraits::recvDispls> recv_displs([[maybe_unused]] new_vector<T> &&) {
    return out_vector_unspecified<T, ptraits::recvDispls>();
}

template<class T>
out_unique_unspecified<T, ptraits::recvDispls> recv_displs([[maybe_unused]] new_pointer<T> &&) {
    return out_unique_unspecified<T, ptraits::recvDispls>();
}

template<class T>
out_vector_alternative<T, ptraits::recvDispls> recv_displs(std::vector<T> &&vec) {
    return out_vector_alternative<T, ptraits::recvDispls>(std::forward<std::vector<T>>(vec));
}

template<class T>
out_unique_unspecified<T, ptraits::recvDispls> recv_displs([[maybe_unused]] T *ptr) {
    return out_unique_unspecified<T, ptraits::recvDispls>();
}


template<class T>
out_vector<T, ptraits::recvCounts> recv_counts(std::vector<T> &vec) {
    return out_vector<T, ptraits::recvCounts>(vec);
}

template<class T>
out_unique<T, ptraits::recvCounts> recv_counts(std::unique_ptr<T[]> &ptr) {
    return out_unique<T, ptraits::recvCounts>(ptr);
}

template<class T>
out_vector_unspecified<T, ptraits::recvCounts> recv_counts([[maybe_unused]] new_vector<T> &&) {
    return out_vector_unspecified<T, ptraits::recvCounts>();
}

template<class T>
out_unique_unspecified<T, ptraits::recvCounts> recv_counts([[maybe_unused]] new_pointer<T> &&) {
    return out_unique_unspecified<T, ptraits::recvCounts>();
}

template<class T>
out_vector_alternative<T, ptraits::recvCounts> recv_counts(std::vector<T> &&vec) {
    return out_vector_alternative<T, ptraits::recvCounts>(std::forward<std::vector<T>>(vec));
}

template<class T>
out_unique_unspecified<T, ptraits::recvCounts> recv_counts([[maybe_unused]] T *ptr) {
    return out_unique_unspecified<T, ptraits::recvCounts>();
}
