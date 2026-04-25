#pragma once

#include "cbow/describer.hh"

namespace cbow {
  struct model {
    matrix_pointer in_matrix;
    matrix_pointer out_matrix;
    std::unique_ptr<std::vector<std::string>> vocabulary;

    describer extract_word_vector_for(const matrix_type &mat) const;
  };
}
