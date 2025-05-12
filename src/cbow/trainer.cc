#include "cbow/trainer.hh"

#include <algorithm>
#include <iostream>
#include <unordered_map>

#include "cbow/loss.hh"
#include "cbow/visitor.hh"
#include "softmax.hh"
#include "topmost.hh"

namespace cbow {
  trainer::trainer(std::deque<std::deque<std::size_t>> &indices, const options &options)
    : m_dimensions(options.dimensions)
    , m_eta(options.eta)
    , m_indices(std::move_iterator(indices.begin()), std::move_iterator(indices.end()))
    , m_verbosity(options.verbosity)
    , m_width(options.width) {
  }

  bool trainer::doesDrawHistogram() const {
    return m_verbosity & 32;
  }

  void trainer::explain(const explain_args &args) const {
    const auto &corpus = args.corpus;
    const auto epoch = args.epoch;
    const auto &indices = args.indices;
    const auto loss = args.loss;
    const auto pos = args.pos;
    const auto &probability = args.probability;

    // header
    std::cerr << std::format("{}/{}/[{}] {{", pos + 1, indices.size(), epoch);

    // sentence with a blank at `pos`
    args.visit(pos, [&corpus, pos](const auto i, const auto j) {
      if (j < static_cast<std::ptrdiff_t>(pos))
        std::cerr << std::format("{} ", i == 0 ? "\x1b[1m<|B|>\x1b[m" : corpus.at(i));
      else if (j == static_cast<std::ptrdiff_t>(pos))
        std::cerr << "\x1b[4m   \x1b[m";
      else
        std::cerr << std::format(" {}", i == 1 ? "\x1b[1m<|E|>\x1b[m" : corpus.at(i));
    });
    std::cerr << '}' << std::endl;

    // candidates
    constexpr int colors[] = {31, 32};
    const auto index = indices.at(pos);
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

  inference_pointer trainer::infer(std::size_t pos, const model &model, const visitor &visit) const {
    // input layer
    auto hidden = vector_type(m_dimensions);
    for (auto i = 0ul; i < m_dimensions; i++) {
      const auto row = model.in_matrix->row(i);
      auto value = static_cast<long double>(0);
      visit(pos, [&row, &value](const auto i) { value += row[i]; });
      hidden[i] = value / (m_width * 2);
    }

    // output layer
    auto probability = *model.out_matrix * hidden;
    softmax(probability);

    return std::make_unique<inference_type>(std::move(hidden), std::move(probability));
  }

  loss_statistics trainer::train(std::size_t epoch, const model &model, std::mt19937_64 &engine) const {
    auto temp = std::vector<std::size_t>(m_indices.size());
    for (auto i = 0zu; i < m_indices.size(); i++)
      temp[i] = i;
    std::shuffle(temp.begin(), temp.end(), engine);
    auto loss = loss_context(m_verbosity);
    for (const auto i : temp) {
      const auto &indices = m_indices.at(i);
      auto inferences = std::unordered_map<std::size_t, vector_type>();
      auto losses = std::unordered_map<std::size_t, element_type>();
      auto positions = std::vector<std::size_t>();
      for (auto i = 0zu; i < indices.size(); i++)
        positions.push_back(i);
      std::shuffle(positions.begin(), positions.end(), engine);
      auto skip = skip_visitor(indices, m_width);
      auto visit = visitor(indices, m_width);
      for (const auto pos : positions) {
        const auto index = indices.at(pos);

        // infer what word is appropriate at `pos`
        auto inference = infer(pos, model, skip);

        // remember the probability in order to explain them later if necessary
        if (m_verbosity & 16)
          inferences[pos] = inference->probability;

        // calculate cross entropy loss and update probability
        auto &prob = inference->probability[index];
        const auto cross_entropy_loss = -std::log(prob);
        prob -= 1;

        // accumulate loss
        losses[pos] = cross_entropy_loss;
        loss.add(cross_entropy_loss);

        // update matrices
        auto hadamard = model.out_matrix->hadamard(inference->probability);
        model.in_matrix->update(m_eta, hadamard, pos, visit, m_width);
        model.out_matrix->update(m_eta, *inference);
      }

      // explain if necessary
      if (m_verbosity & 16) {
#if 0 // change to 1 if you don't mind the order of inference
        for (const auto [pos, prob] : inferences) {
          auto args = explain_args{
            .corpus = *model.corpus,
            .epoch = epoch,
            .indices = indices,
            .loss = losses.at(pos),
            .pos = pos,
            .probability = prob,
            .visit = visit,
          };
          explain(args);
        }
#else
        for (auto i = 0zu; i < indices.size(); i++) {
          auto args = explain_args{
            .corpus = *model.corpus,
            .epoch = epoch,
            .indices = indices,
            .loss = losses.at(i),
            .pos = i,
            .probability = inferences.find(i)->second,
            .visit = visit,
          };
          explain(args);
        }
#endif
      }
    }
    return loss.statistics();
  }
}
