#pragma once

#include <string>

#include "cbow.hh"

namespace cbow {
  struct loss_statistics;

  struct loss_context {
  private:
    std::vector<element_type> m_list;
    long double m_total;
    const int m_verbosity;

  public:
    loss_context() = delete;
    loss_context(int verbosity);

    void add(element_type error);

    loss_statistics statistics();
  };

  struct loss_statistics {
    long double average;
    std::string hist;
    long double max;
    long double median;
    long double min;
    long double variance;
  };
}
