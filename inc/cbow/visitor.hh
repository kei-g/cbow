#pragma once

#include <functional>

#include "cbow.hh"

namespace cbow {
  struct visitor {
    typedef std::ptrdiff_t index_type;
    typedef std::size_t pos_type;
    typedef const std::function<void(pos_type, index_type)> &const_func_ref;

  protected:
    const_indices_ref m_indices;
    const unsigned long m_width;

    void visit(bool &end, index_type j, const_func_ref fn) const;

  public:
    visitor(const_indices_ref indices, unsigned long width);

    virtual void operator()(pos_type pos, const_func_ref fn) const;
    virtual void operator()(pos_type pos, const std::function<void(pos_type)> &fn) const;
  };

  struct skip_visitor : visitor {
    skip_visitor(const_indices_ref indices, unsigned long width);

    void operator()(pos_type pos, const_func_ref fn) const override;
  };
}
