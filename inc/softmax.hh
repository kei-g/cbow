#pragma once

#include <cmath>
#include <concepts>

/**
 * @brief Normalizes the input vector into a probability distribution where elements sum to 1. Performed in-place.
 *
 * @tparam T Floating-point type (float, double) for calculations.
 * @param target Vector of values to normalize.
 */
template <std::floating_point T>
void softmax(std::vector<T> &target) {
  auto sum = static_cast<T>(0);
  for (auto &value : target)
    sum += value = std::exp(value);
  for (auto &value : target)
    value /= sum;
}
