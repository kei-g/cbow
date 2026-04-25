#include <iostream>
#include <limits>

#include "cbow/trainer.hh"

static cbow::trainer load(int argc, char *argv[], std::mt19937_64 &engine, cbow::model *model);

int main(int argc, char *argv[]) {
  auto dev = std::random_device();
  auto engine = std::mt19937_64(dev());
  cbow::model model;
  auto trainer = load(argc, argv, engine, &model);
  if (trainer.doesDrawHistogram())
    for (auto i = 0; i < 4; i++)
      std::cerr << std::endl;

  cbow::matrix_pointer best_mat;
  auto best_epoch = 0zu;
  auto best_loss = std::numeric_limits<long double>::infinity();
  auto context = signal();
  for (auto epoch = 0zu; !context.interrupted(); epoch++) {
    const auto loss = trainer.train(epoch, context);
    if (trainer.doesDrawHistogram()) {
      std::cerr << "\x1b[4A";
      std::cerr << loss.hist << std::endl;
    }
    std::cerr << std::format("[{}] 損失: ", epoch);
    std::cerr << '{';
    std::cerr << std::format("平均: {:.4f}, 最小: {:.4f}, 最大: {:.4f}, ", loss.average, loss.min, loss.max);
    std::cerr << std::format("標準偏差: {:.4f}", std::sqrt(loss.variance));
    std::cerr << '}' << std::endl;
    if (loss.average <= best_loss) {
      best_epoch = epoch;
      best_loss = loss.average;
      model.in_matrix->copy_to(best_mat);
    }
    else if (best_loss + static_cast<long double>(1) / 128 < loss.average) {
      std::cerr << std::format("[{}] 損失: {}", best_epoch, best_loss) << std::endl;
      break;
    }
  }

  // output word vectors
  std::cout << model.extract_word_vector_for(*best_mat);

  return 0;
}

#include <fcntl.h>
#include <unistd.h>

#include "corpus_manager.hh"

static cbow::trainer load(int argc, char *argv[], std::mt19937_64 &engine, cbow::model *model) {
  const auto opts = cbow::options::parse(argc, argv);
  auto tr = ::corpus_manager(opts);
  extern int optind;
  if (optind < argc) {
    auto fd = open(argv[optind], O_RDONLY);
    if (fd < 0)
      throw;
    tr.load(fd);
  }
  else
    tr.load(0);

  // exclude infrequent words from training
  tr.prune(opts.min_count);

  // describe statistics of corpus and vocabulary
  tr.describe_to(std::cerr);

  // populate a CBoW model and trainer
  return tr.populate(model, engine);
}
