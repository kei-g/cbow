#include "cbow/model.hh"

namespace cbow {
  describer model::extract_word_vector_for(const matrix_type &mat) const {
    return describer(*vocabulary, mat);
  }
}
