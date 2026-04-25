#include "cbow/options.hh"

#include <getopt.h>

#include <string>

namespace cbow {
  options options::parse(int argc, char *argv[]) {
    auto r = cbow::options();
    struct option opts[] = {
      {"case-sensitive", no_argument, nullptr, 'c'}, {"dimension", required_argument, nullptr, 'd'},
      {"eta", required_argument, NULL, 'e'},         {"min-count", required_argument, nullptr, 'm'},
      {"verbosity", required_argument, nullptr, 'v'}, {"width", required_argument, nullptr, 'w'},
      {NULL, 0, NULL, 0},
    };
    for (;;) {
      auto i = 0, c = getopt_long(argc, argv, "cd:e:m:v:w:", opts, &i);
      if (c < 0)
        break;
      if (optopt)
        throw;
      switch (c) {
      case 'c':
        r.case_sensitive = true;
        break;
      case 'd':
        r.dimensions = std::stoul(optarg);
        break;
      case 'e':
        r.eta = std::stold(optarg);
        break;
      case 'm':
        r.min_count = std::stoul(optarg);
        break;
      case 'v':
        r.verbosity = std::stoi(optarg);
        break;
      case 'w':
        r.width = std::stoul(optarg);
        break;
      }
    }
    return r;
  }

  options::options()
    : case_sensitive(false)
    , dimensions(128)
    , eta(1.0 / 512)
    , min_count(5)
    , verbosity(0)
    , width(4) {
  }
}
