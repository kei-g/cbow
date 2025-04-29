#pragma once

#include "cbow/describer.hh"

namespace cbow {
  struct model {
    std::unique_ptr<std::vector<std::string>> corpus;
    matrix_pointer in_matrix;
    matrix_pointer out_matrix;

    describer extract_word_vector_for(const matrix_type &mat) const;
  };
}
