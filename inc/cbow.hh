#pragma once

#include <cstddef>
#include <deque>
#include <vector>

#include "matrix.hh"

namespace cbow {
  typedef std::deque<std::size_t> indices_type;
  typedef const indices_type &const_indices_ref;

  typedef double element_type;
  typedef ::matrix<element_type> matrix_type;
  typedef matrix_type &matrix_ref;
  typedef std::unique_ptr<matrix_type> matrix_pointer;
  typedef std::vector<element_type> vector_type;

  typedef std::size_t text_position_type;
  typedef std::vector<text_position_type> text_positions;
  typedef std::size_t vocabulary_position_type;
}
