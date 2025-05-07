#include "cbow/describer.hh"

std::ostream &cbow::operator<<(std::ostream &dest, const cbow::describer &desc) {
  desc.describe_to(dest);
  return dest;
}

namespace cbow {
  describer::describer(const std::vector<std::string> &corpus, const matrix_type &mat)
    : m_corpus(corpus)
    , m_mat(mat.transpose()) {
  }

  void describer::describe_to(std::ostream &dest) const {
    dest << '{' << std::endl;
    for (auto i = 2zu; i < m_corpus.size(); i++) {
      const auto row = m_mat->row(i);
      const auto &word = m_corpus.at(i);
      dest << std::format("  {:?}: [", word) << std::endl;
      for (auto j = 0zu; j < m_mat->columns; j++) {
        dest << std::format("    {:.16f}", row[j]);
        if (j + 1 < m_mat->columns)
          dest << ',';
        dest << std::endl;
      }
      dest << "  ]";
      if (i + 1 < m_corpus.size())
        dest << ',';
      dest << std::endl;
    }
    dest << '}' << std::endl;
  }
}
