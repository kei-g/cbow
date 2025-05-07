#include <iostream>
#include <limits>

#include "cbow/trainer.hh"

static cbow::trainer load(int argc, char *argv[], std::mt19937_64 &engine, cbow::model *model);

int main(int argc, char *argv[]) {
  auto dev = std::random_device();
  auto engine = std::mt19937_64(dev());
  cbow::model model;
  auto trainer = load(argc, argv, engine, &model);

  cbow::matrix_pointer best_mat;
  auto best_epoch = 0zu;
  auto best_loss = std::numeric_limits<long double>::infinity();
  for (auto epoch = 0zu;; epoch++) {
    const auto loss = trainer.train(epoch, model, engine);
    std::cerr << std::format("[{}] 損失: {}", epoch, loss) << std::endl;
    if (loss <= best_loss) {
      best_epoch = epoch;
      best_loss = loss;
      model.in_matrix->copy_to(best_mat);
    }
    else if (best_loss + 1 < loss) {
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

#include "text_reader.hh"

static cbow::trainer load(int argc, char *argv[], std::mt19937_64 &engine, cbow::model *model) {
  const auto opts = cbow::options::parse(argc, argv);
  auto tr = ::text_reader(opts);
  extern int optind;
  if (optind < argc) {
    auto fd = open(argv[optind], O_RDONLY);
    if (fd < 0)
      throw;
    tr.load(fd);
  }
  else
    tr.load(0);

  // describe corpus and words if necessary
  tr.describe_to(std::cerr);

  // populate a CBoW model and trainer
  return tr.populate(model, engine);
}
