#pragma once

#include <cmath>
#include <concepts>

template <std::floating_point T>
void softmax(std::vector<T> &target) {
  auto sum = static_cast<T>(0);
  for (auto &value : target)
    sum += value = std::exp(value);
  for (auto &value : target)
    value /= sum;
}
