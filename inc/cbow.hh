#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "matrix.hh"

namespace cbow {
  typedef double element_type;
  typedef ::matrix<element_type> matrix_type;
  typedef matrix_type &matrix_ref;
  typedef std::unique_ptr<matrix_type> matrix_pointer;
  typedef std::vector<element_type> vector_type;

  typedef std::size_t text_position_type;
  typedef std::vector<text_position_type> text_positions;
  typedef std::size_t vocabulary_position_type;

  struct inference_type {
    vector_type hidden;
    vector_type probability;

    inference_type(vector_type &&hidden, vector_type &&probability);
  };

  typedef std::unique_ptr<inference_type> inference_pointer;
}
