#pragma once

#include "file_descriptor.hh"

namespace utf8 {
  struct reader {
    typedef file_descriptor::const_functor_ref const_functor_ref;

    void accept(file_descriptor &fd, const_functor_ref functor) const;
  };
}
