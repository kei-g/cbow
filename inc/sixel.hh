#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace sixel {
  std::string compress(const std::string &sixel);

  std::string create(const std::vector<std::size_t> &hist, const std::size_t average_index, const int height);
}
