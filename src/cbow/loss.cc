#include "cbow/loss.hh"

#include <algorithm>

#include "sixel.hh"

namespace cbow {
  loss_context::loss_context(int verbosity)
    : m_total(0)
    , m_verbosity(verbosity) {
  }

  void loss_context::add(element_type error) {
    m_list.push_back(error);
    m_total += error;
  }

  loss_statistics loss_context::statistics() {
    std::sort(m_list.begin(), m_list.end(), [](const auto lhs, const auto rhs) { return lhs < rhs; });

    if (m_verbosity & 32) {
      constexpr auto hist_res = 256zu; // resolution of the histogram, equivalent to width of sixel in pixels.
      constexpr auto minimum_loss_width = static_cast<element_type>(8); // minimum loss to be drawn to sixel
      constexpr auto sixel_height = 54;                                 // height of sixel in pixels

      const auto average = m_total / m_list.size();                        // average of losses
      const auto loss_width = std::max(minimum_loss_width, m_list.back()); // maximum loss to be drawn to sixel
      auto hist = std::vector<std::size_t>(hist_res);                      // histogram of losses fit to the `hist_res`
      auto variance = static_cast<long double>(0);                         // variance of losses
      for (const auto value : m_list) {
        const auto index = std::min(static_cast<std::size_t>(value * hist_res / loss_width), hist_res - 1);
        hist[index]++;
        const auto difference = value - average;
        variance += difference * difference;
      }
      variance /= m_list.size();

      auto peak = 0zu; // most frequency in the histogram of losses
      for (const auto freq : hist)
        if (peak < freq)
          peak = freq;
      while (std::gcd(peak, sixel_height) != 1)
        peak++;
      for (auto &freq : hist) {
        freq *= sixel_height;
        freq /= peak;
      }

      return loss_statistics{
        .average = average,
        .hist = sixel::create(hist, static_cast<std::size_t>(average * hist_res / loss_width), sixel_height),
        .max = m_list.back(),
        .median = m_list.at(m_list.size() / 2),
        .min = m_list.front(),
        .variance = variance,
      };
    }
    else {
      const auto average = m_total / m_list.size(); // average of losses
      auto variance = static_cast<long double>(0);  // variance of losses
      for (const auto value : m_list) {
        const auto difference = value - average;
        variance += difference * difference;
      }
      variance /= m_list.size();

      return loss_statistics{
        .average = average,
        .hist = "",
        .max = m_list.back(),
        .median = m_list.at(m_list.size() / 2),
        .min = m_list.front(),
        .variance = variance,
      };
    }
  }
}
