#pragma once

namespace cbow {
  struct options {
    static options parse(int argc, char *argv[]);

    bool case_sensitive;
    unsigned long dimensions;
    long double eta;
    unsigned long min_count;
    int verbosity;
    unsigned long width;

    options();
  };
}
