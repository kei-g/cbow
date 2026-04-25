#include "cbow/trainer.hh"

#include <algorithm>
#include <iostream>

#include "cbow/loss.hh"
#include "cbow/visitor.hh"
#include "softmax.hh"
#include "topmost.hh"

namespace cbow {
  trainer::trainer(std::mt19937_64 &engine, std::deque<std::deque<std::size_t>> &indices, const model &model,
                   const options &options)
    : m_dimensions(options.dimensions)
    , m_engine(engine)
    , m_eta(options.eta)
    , m_indices(std::move_iterator(indices.begin()), std::move_iterator(indices.end()))
    , m_model(model)
    , m_total_tokens(0)
    , m_verbosity(options.verbosity)
    , m_width(options.width) {
    for (const auto &sentence : m_indices)
      m_total_tokens += sentence.size();
  }

  bool trainer::doesDrawHistogram() const {
    return m_verbosity & 32;
  }

  void trainer::explain(const explain_args &args) const {
    const auto epoch = args.epoch;
    const auto &indices = args.indices;
    const auto loss = args.loss;
    const auto pos = args.pos;
    const auto &probability = args.probability;
    const auto &vocabulary = args.vocabulary;

    // header
    std::cerr << std::format("{}/{}/[{}] {{", pos + 1, indices.size(), epoch);

    // sentence with a blank at `pos`
    args.visit(pos, [pos, &vocabulary](const auto i, const auto j) {
      if (j < static_cast<std::ptrdiff_t>(pos))
        std::cerr << std::format("{} ", i == 0 ? "\x1b[1m<|B|>\x1b[m" : vocabulary.at(i));
      else if (j == static_cast<std::ptrdiff_t>(pos))
        std::cerr << "\x1b[4m   \x1b[m";
      else
        std::cerr << std::format(" {}", i == 1 ? "\x1b[1m<|E|>\x1b[m" : vocabulary.at(i));
    });
    std::cerr << '}' << std::endl;

    // candidates
    constexpr int colors[] = {31, 32};
    const auto index = indices.at(pos);
    auto variance = static_cast<element_type>(0);
    for (const auto &[prob, idx] : topmost(3, probability, &variance)) {
      const auto correct = idx == index;
      const auto color = colors[static_cast<std::size_t>(correct)];
      const auto &word = vocabulary.at(idx);
      std::cerr << std::format("    \x1b[{}m{}\x1b[m: {:.4f}%", color, word, prob * 100) << std::endl;
    }

    // statistics of candidates
    std::cerr << std::format("      {{分散: {:.9f}, 標準偏差: {:.9f}}}", variance, std::sqrt(variance)) << std::endl;

    // correct label and its current probability
    const auto prob = probability.at(index) * 100;
    const auto &word = vocabulary.at(index);
    std::cerr << std::format("   (\x1b[32m{}\x1b[m: \x1b[33m{:.4f}\x1b[m%) 損失: {}", word, prob, loss) << std::endl;
  }

  void trainer::infer(std::size_t pos, const visitor &visit, vector_type &hidden, vector_type &probability) const {
    // input layer
    for (auto i = 0ul; i < m_dimensions; i++) {
      const auto row = m_model.in_matrix->row(i);
      auto value = static_cast<long double>(0);
      visit(pos, [&row, &value](const auto i) { value += row[i]; });
      hidden[i] = value / (m_width * 2);
    }

    // output layer
    m_model.out_matrix->multiply(hidden, probability);
    softmax(probability);
  }

  loss_statistics trainer::train(std::size_t epoch, signal &ctx) const {
    auto max_length = 0zu;
    auto hidden = vector_type(m_dimensions);
    auto hidden_gradient = vector_type(m_dimensions);
    auto loss = loss_context(m_verbosity, m_total_tokens);
    auto probability = vector_type(m_model.out_matrix->rows);
    auto shuffled_indices = text_positions(m_indices.size());
    for (auto j = 0zu; j < m_indices.size(); j++) {
      shuffled_indices[j] = j;
      const auto size = m_indices[j].size();
      if (max_length < size)
        max_length = size;
    }
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), m_engine);
    auto inferences = std::vector<vector_type>(max_length);
    auto losses = vector_type(max_length);
    auto shuffled_positions = text_positions(max_length);
    for (const auto i : shuffled_indices) {
      auto interrupted = ctx.interrupted();
      if (interrupted)
        break;
      const auto &indices = m_indices.at(i);
      for (auto j = 0zu; j < indices.size(); j++)
        shuffled_positions[j] = j;
      std::shuffle(shuffled_positions.begin(), shuffled_positions.begin() + indices.size(), m_engine);
      auto skip = skip_visitor(indices, m_width);
      auto visit = visitor(indices, m_width);
      for (auto j = 0zu; j < indices.size(); j++) {
        const auto pos = shuffled_positions[j];
        interrupted = ctx.interrupted();
        if (interrupted)
          break;
        const auto index = indices.at(pos);

        // infer what word is appropriate at `pos`
        infer(pos, skip, hidden, probability);

        // remember the probability in order to explain them later if necessary
        if (m_verbosity & 16)
          inferences[pos] = probability;

        // calculate cross entropy loss and update probability
        auto &prob = probability[index];
        const auto cross_entropy_loss = -std::log(prob);
        prob -= 1;

        // accumulate loss
        losses[pos] = cross_entropy_loss;
        loss.add(cross_entropy_loss);

        // update matrices
        m_model.out_matrix->multiply_transpose(probability, hidden_gradient);
        m_model.in_matrix->update_embedding(m_eta, hidden_gradient, pos, visit, m_width);
        m_model.out_matrix->update_output(m_eta, hidden, probability);
      }
      if (interrupted)
        break;

      // explain if necessary
      if (m_verbosity & 16) {
#if 0 // change to 1 if you don't mind the order of inference
        for (const auto [pos, prob] : inferences) {
          auto args = explain_args{
            .epoch = epoch,
            .indices = indices,
            .loss = losses[pos],
            .pos = pos,
            .probability = prob,
            .visit = visit,
            .vocabulary = *m_model.vocabulary,
          };
          explain(args);
        }
#else
        for (auto j = 0zu; j < indices.size(); j++) {
          auto args = explain_args{
            .epoch = epoch,
            .indices = indices,
            .loss = losses[j],
            .pos = j,
            .probability = inferences[j],
            .visit = visit,
            .vocabulary = *m_model.vocabulary,
          };
          explain(args);
        }
#endif
      }
    }
    return loss.statistics();
  }
}
