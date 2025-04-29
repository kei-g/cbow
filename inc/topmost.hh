#pragma once

#include <concepts>
#include <list>
#include <vector>

template <typename T, typename U>
concept VarianceComputableWith =
  requires(const U &lhs, const T &rhs) { lhs - rhs; } && requires(const T &lhs, const T &rhs) { lhs * rhs; } &&
  requires(const T &lhs, const std::size_t rhs) { lhs / rhs; } && requires(T &lhs, const T &rhs) { lhs += rhs; } &&
  requires(T &lhs, const std::size_t rhs) { lhs /= rhs; };

template <std::totally_ordered T, VarianceComputableWith<T> U>
auto topmost(const std::size_t num, const std::vector<T> &source, U *variance) {
  const auto count = source.size();
  const auto average = static_cast<U>(1) / count;
  auto result = std::list<std::pair<T, std::size_t>>();
  for (auto index = 0zu; index < count; index++) {
    const auto &elem = source.at(index);
    if (result.empty())
      result.push_back(std::make_pair(elem, index));
    else {
      for (auto iter = result.begin(); iter != result.end(); iter++)
        if (iter->first <= elem) {
          result.insert(iter, std::make_pair(elem, index));
          break;
        }
      if (num < result.size())
        result.pop_back();
    }
    const auto diff = elem - average;
    *variance += diff * diff;
  }
  *variance /= count;
  return result;
}
