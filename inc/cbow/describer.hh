#pragma once

#include <ostream>

#include "cbow.hh"

namespace cbow {
  struct describer {
    friend std::ostream &operator<<(std::ostream &dest, const describer &desc);

  private:
    const matrix_pointer m_mat;
    const std::vector<std::string> &m_vocabulary;

    void describe_to(std::ostream &dest) const;

  public:
    describer(const std::vector<std::string> &vocabulary, const matrix_type &mat);
  };
}
