#pragma once

#include <deque>

struct text_reader;

#include "cbow.hh"
#include "cbow/describer.hh"
#include "cbow/model.hh"
#include "cbow/options.hh"

namespace cbow {
  struct trainer {
    friend struct ::text_reader;

  private:
    const unsigned long m_dimensions;
    const long double m_eta;
    const text_positions m_indices;
    const int m_verbosity;
    const unsigned long m_width;

    struct explain_args {
      const std::vector<std::string> &corpus;
      std::size_t epoch;
      element_type loss;
      text_position_type pos;
      const vector_type &probability;
    };
    void explain(const explain_args &args) const;

    inference_pointer infer(text_position_type position, const model &model) const;

    /**
     * Constructor
     */
    trainer(std::deque<std::size_t> &indices, const options &options);

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
