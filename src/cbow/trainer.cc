#include "cbow/trainer.hh"

#include <algorithm>
#include <iostream>
#include <unordered_map>

#include "softmax.hh"
#include "topmost.hh"

namespace cbow {
  trainer::trainer(std::deque<std::size_t> &indices, const options &options)
    : m_dimensions(options.dimensions)
    , m_eta(options.eta)
    , m_indices(std::move_iterator(indices.begin()), std::move_iterator(indices.end()))
    , m_verbosity(options.verbosity)
    , m_width(options.width) {
  }

  void trainer::explain(const explain_args &args) const {
    const auto &corpus = args.corpus;
    const auto epoch = args.epoch;
    const auto loss = args.loss;
    const auto pos = args.pos;
    const auto probability = args.probability;

    // header
    std::cerr << std::format("{}/{}/[{}] {{", pos + 1, m_indices.size() - m_width, epoch);

    // sentence with a blank at `pos`
    for (auto j = pos - m_width; j < pos; j++)
      std::cerr << std::format("{} ", corpus.at(m_indices.at(j)));
    std::cerr << "\x1b[4m   \x1b[m";
    for (auto j = pos + 1; j <= pos + m_width; j++)
      std::cerr << std::format(" {}", corpus.at(m_indices.at(j)));
    std::cerr << '}' << std::endl;

    // candidates
    constexpr int colors[] = {31, 32};
    const auto index = m_indices.at(pos);
    auto variance = static_cast<element_type>(0);
    for (const auto &[prob, idx] : topmost(3, probability, &variance)) {
      const auto correct = idx == index;
      const auto color = colors[static_cast<std::size_t>(correct)];
      const auto &word = corpus.at(idx);
      std::cerr << std::format("    \x1b[{}m{}\x1b[m: {:.4f}%", color, word, prob * 100) << std::endl;
    }

    // statistics of candidates
    std::cerr << std::format("      {{分散: {:.9f}, 標準偏差: {:.9f}}}", variance, std::sqrt(variance)) << std::endl;

    // correct label and its current probability
    const auto &word = corpus.at(index);
    const auto prob = probability.at(index) * 100;
    std::cerr << std::format("   (\x1b[32m{}\x1b[m: \x1b[33m{:.4f}\x1b[m%) 損失: {}", word, prob, loss) << std::endl;
  }

  inference_pointer trainer::infer(text_position_type position, const cbow::model &model) const {
    // input layer
    auto hidden = vector_type(m_dimensions);
    for (auto i = 0ul; i < m_dimensions; i++) {
      const auto row = model.in_matrix->row(i);
      auto value = static_cast<long double>(0);
      for (auto j = static_cast<text_position_type>(position - m_width); j < position - 1; j++)
        value += row[m_indices.at(j)];
      for (auto j = static_cast<text_position_type>(position + 1); j < position + m_width; j++)
        value += row[m_indices.at(j)];
      hidden[i] = value / (m_width * 2);
    }

    // output layer
    auto probability = *model.out_matrix * hidden;
    softmax(probability);

    return std::make_unique<inference_type>(std::move(hidden), std::move(probability));
  }

  long double trainer::train(std::size_t epoch, const model &model, std::mt19937_64 &engine) const {
    // shuffle the training data
    auto positions = std::vector<text_position_type>();
    for (auto position = static_cast<text_position_type>(m_width); position < m_indices.size() - m_width; position++)
      positions.push_back(position);
    std::shuffle(positions.begin(), positions.end(), engine);

    // train
    auto losses = vector_type(positions.size());
    auto loss = static_cast<long double>(0);
    auto inferences = std::unordered_map<text_position_type, vector_type>();
    for (const auto position : positions) {
      const auto index = m_indices.at(position);

      // infer what word is appropriate at `position`
      auto inference = infer(position, model);

      // remember the probability in order to explain them later if necessary
      if (2 < m_verbosity)
        inferences[position] = inference->probability;

      // calculate cross entropy loss and update probability
      auto &prob = inference->probability[index];
      const auto cross_entropy_loss = -std::log(prob);
      prob -= 1;

      // accumulate loss
      losses[position - m_width] = cross_entropy_loss;
      loss += cross_entropy_loss;

      // update matrices
      auto hadamard = *model.out_matrix->transpose() * inference->probability;
      model.in_matrix->update(m_eta, hadamard, m_indices, position, m_width);
      model.out_matrix->update(m_eta, *inference);
    }

    // explain if necessary
    if (2 < m_verbosity)
#if 0 // change to 1 if you don't mind the order of inference
      for (const auto [pos, prob] : inferences) {
        auto args = explain_args {
          .corpus = *model.corpus,
          .epoch = epoch,
          .loss = losses.at(pos - m_width),
          .pos = pos,
          .probability = prob,
        };
        explain(args);
      }
#else
      for (auto pos = static_cast<text_position_type>(m_width); pos < m_indices.size() - m_width; pos++) {
        auto args = explain_args{
          .corpus = *model.corpus,
          .epoch = epoch,
          .loss = losses.at(pos - m_width),
          .pos = pos,
          .probability = inferences.find(pos)->second,
        };
        explain(args);
      }
#endif

    return loss;
  }
}
