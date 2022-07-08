//
// Created by hao on 06/07/22.
//

#ifndef CUTLASSTEST_INTERVAL_CUH
#define CUTLASSTEST_INTERVAL_CUH

#include <cuda.h>

template<typename T>
struct Interval {
    T lower = 0.; /// lower bound

    T upper = 0.; /// upper bound

    /// constructors
    __device__ __host__ Interval() = default;

    __device__ __host__ Interval(const T &lower, const T &upper) : lower(lower), upper(upper) {}

    __device__ __host__ Interval(const T &num) : lower(num), upper(num) {}

    /// Multiplication, use template-based simulated dynamic binding instead of inheritance
    __device__ Interval<T> operator*(const Interval<T> &rhs) const;
    /// Multiplication with a number only
    __device__ Interval<T> operator*(const T &rhs) const {
        return *this * Interval<T>(rhs);
    }

    /// Inplace multiplication, delegate to operator *
    __device__ Interval<T> &operator*=(const Interval<T> &rhs) {
        *this = *this * rhs;
        return *this;
    }
    /// Inplace multiplication with number only, delegate to operator *=
    __device__ Interval<T> &operator*=(const T &rhs) {
        *this = *this * rhs;
        return *this;
    }

    /// Addition, use template-based simulated dynamic binding instead of inheritance
    __device__ Interval<T> operator+(const Interval<T> &rhs) const;
    /// Addition with a number only
    __device__ Interval<T> operator+(const T &rhs) const {
        return *this + Interval<T>(rhs);
    }

    /// Inplace addition, delegate to operator +
    __device__ Interval<T> &operator+=(const Interval<T> &rhs) {
        *this = *this + rhs;
        return *this;
    }
    /// Inplace addition with number only, delegate to operator +=
    __device__ Interval<T> &operator+=(const T &rhs) {
        *this = *this + rhs;
        return *this;
    }

    /// Test if two intervals are equivalent
    __device__ bool operator==(const Interval<T> &rhs) const {
        return (lower == rhs.lower) && (upper == rhs.upper);
    }

    /// Test if two intervals are different
    __device__ bool operator!=(const Interval<T> &rhs) const {
        return !(*this == rhs);
    }

    __device__ __host__ Interval<T> &operator=(const int &rhs) {
        lower = rhs;
        upper = rhs;
        return *this;
    }
    __device__ __host__ Interval<T> &operator=(const double &src) {
        lower = src;
        upper = src;
        return *this;
    }

    __device__ __host__ Interval<T> &operator=(const float &src) {
        lower = src;
        upper = src;
        return *this;
    }

    __device__ __host__ Interval<T> &operator=(const Interval<T> &src) {
        lower = src.lower;
        upper = src.upper;
        return *this;
    }

    /// copy constructor
    __device__ __host__ Interval<T>(const Interval<T> &src) {
        *this = src;
    }

};

/// overloading <<


template<>
__device__ Interval<double> Interval<double>::operator*(const Interval<double> &rhs) const {
    return {fmin(fmin(__dmul_rd(this->lower, rhs.lower), __dmul_rd(this->lower, rhs.upper)),
                 fmin(__dmul_rd(this->upper, rhs.lower), __dmul_rd(this->upper, rhs.upper))),
            fmax(fmax(__dmul_ru(this->lower, rhs.lower), __dmul_ru(this->lower, rhs.upper)),
                 fmax(__dmul_ru(this->upper, rhs.lower), __dmul_ru(this->upper, rhs.upper)))};
}

template<>
__device__ Interval<float> Interval<float>::operator*(const Interval<float> &rhs) const {
    return {fminf(fminf(__fmul_rd(this->lower, rhs.lower), __fmul_rd(this->lower, rhs.upper)),
                  fminf(__fmul_rd(this->upper, rhs.lower), __fmul_rd(this->upper, rhs.upper))),
            fmaxf(fmaxf(__fmul_ru(this->lower, rhs.lower), __fmul_ru(this->lower, rhs.upper)),
                  fmaxf(__fmul_ru(this->upper, rhs.lower), __fmul_ru(this->upper, rhs.upper)))};
}

template<>
__device__ Interval<float> Interval<float>::operator+(const Interval<float> &rhs) const {
    return {__fadd_rd(this->lower, rhs.lower),
            __fadd_ru(this->upper, rhs.upper)};
}

template<>
__device__ Interval<double> Interval<double>::operator+(const Interval<double> &rhs) const {
    return {__dadd_rd(this->lower, rhs.lower),
            __dadd_ru(this->upper, rhs.upper)};
}

#endif //CUTLASSTEST_INTERVAL_CUH
