#pragma once

#include <deque>

struct text_reader;

#include "cbow.hh"
#include "cbow/describer.hh"
#include "cbow/model.hh"
#include "cbow/options.hh"
#include "cbow/visitor.hh"

namespace cbow {
  struct trainer {
    friend struct ::text_reader;

  private:
    const unsigned long m_dimensions;
    const long double m_eta;
    const std::vector<std::deque<std::size_t>> m_indices;
    const int m_verbosity;
    const unsigned long m_width;

    struct explain_args {
      const std::vector<std::string> &corpus;
      std::size_t epoch;
      const std::deque<std::size_t> &indices;
      element_type loss;
      text_position_type pos;
      const vector_type &probability;
      const visitor &visit;
    };
    void explain(const explain_args &args) const;

    inference_pointer infer(std::size_t pos, const model &model, const visitor &visit) const;

    /**
     * Constructor
     */
    trainer(std::deque<std::deque<std::size_t>> &indices, const options &options);

  public:
    trainer() = delete;
    trainer(const trainer &) = delete;
    trainer(trainer &&) = delete;

    /**
     * Train
     */
    long double train(std::size_t epoch, const model &model, std::mt19937_64 &engine) const;
  };
}
