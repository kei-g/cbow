#include "sixel.hh"

#include <algorithm>
#include <deque>
#include <regex>
#include <sstream>

namespace sixel {
  std::string compress(const std::string &sixel) {
    auto replaced = std::regex_replace(sixel, std::regex("[\x3f-\x7e]{3,}"), " $& ");
    if (sixel == replaced)
      return sixel;
    auto source = std::stringstream(replaced);
    auto temp = std::deque<std::string>();
    for (;;) {
      auto item = std::string();
      source >> item;
      if (item.empty())
        break;
      const auto c = item.back();
      if (std::all_of(item.cbegin(), item.cend(), [c](const auto ch) { return ch == c; })) {
        auto dest = std::stringstream();
        dest << '!';
        dest << item.size();
        dest << c;
        temp.push_back(dest.str());
      }
      else
        temp.push_back(item);
    }
    return std::ranges::fold_left(temp, std::string(), [](auto acc, auto value) { return acc + value; });
  }

  std::string create(const std::vector<std::size_t> &hist, const std::size_t average_index, const int height) {
    auto sixel = std::stringstream();
    sixel << "\x1bP;0;0;q#0;2;9;9;9#1;2;77;77;77#2;2;0;33;0#3;2;0;88;0";
    const auto put = [average_index, &hist, &sixel](int c1, int c2, int i, const char pattern[], char c3) {
      sixel << '#';
      sixel << c1;
      for (auto j = 0zu; j < hist.size(); j++) {
        if (j == average_index) {
          sixel << '#';
          sixel << c2;
        }
        const auto c = static_cast<int>(hist[j]);
        sixel << pattern[std::min(std::max(0, c - i), 6)];
        if (j == average_index) {
          sixel << '#';
          sixel << c1;
        }
      }
      sixel << c3;
    };
    const char pattern0[] = {'~', '^', 'N', 'F', 'B', '@', '?'};
    const char pattern1[] = {'?', '_', 'o', 'w', '{', '}', '~'};
    for (auto i = height - 6; 0 <= i; i -= 6) {
      put(0, 2, i, pattern0, '$');
      put(1, 3, i, pattern1, '-');
    }
    sixel << "\x1b\\";
    return compress(sixel.str());
  }
}
