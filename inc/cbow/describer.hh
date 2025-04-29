#pragma once

#include <ostream>

#include "cbow.hh"

namespace cbow {
  struct describer {
    friend std::ostream &operator<<(std::ostream &dest, const describer &desc);

  private:
    const std::vector<std::string> &m_corpus;
    const matrix_pointer m_mat;

    void describe_to(std::ostream &dest) const;

  public:
    describer(const std::vector<std::string> &corpus, const matrix_type &mat);
  };
}
