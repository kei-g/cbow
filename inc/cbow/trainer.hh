#pragma once

#include <deque>

struct corpus_manager;

#include "cbow.hh"
#include "cbow/describer.hh"
#include "cbow/loss.hh"
#include "cbow/model.hh"
#include "cbow/options.hh"
#include "cbow/visitor.hh"
#include "signal.hh"

namespace cbow {
  struct trainer {
    friend struct ::corpus_manager;

  private:
    const unsigned long m_dimensions;
    std::mt19937_64 &m_engine;
    const long double m_eta;
    const std::vector<std::deque<std::size_t>> m_indices;
    const model &m_model;
    const int m_verbosity;
    const unsigned long m_width;

    struct explain_args {
      std::size_t epoch;
      const std::deque<std::size_t> &indices;
      element_type loss;
      text_position_type pos;
      const vector_type &probability;
      const visitor &visit;
      const std::vector<std::string> &vocabulary;
    };
    void explain(const explain_args &args) const;

    inference_pointer infer(std::size_t pos, const visitor &visit) const;

    /**
     * Constructor
     */
    trainer(std::mt19937_64 &engine, std::deque<std::deque<std::size_t>> &indices, const model &model,
            const options &options);

  public:
    trainer() = delete;
    trainer(const trainer &) = delete;
    trainer(trainer &&) = delete;

    bool doesDrawHistogram() const;

    /**
     * Train
     */
    loss_statistics train(std::size_t epoch, signal &context) const;
  };
}
