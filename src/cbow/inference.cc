#include "cbow.hh"

namespace cbow {
  inference_type::inference_type(vector_type &&hidden, vector_type &&probability)
    : hidden(hidden)
    , probability(probability) {
  }
}
