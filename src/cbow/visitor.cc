#include "cbow/visitor.hh"

namespace cbow {
  visitor::visitor(const_indices_ref indices, unsigned long width)
    : m_indices(indices)
    , m_width(width) {
  }

  void visitor::visit(bool &end, index_type j, const_func_ref fn) const {
    if (j < 0)
      fn(0zu, j);
    else if (end)
      fn(1zu, j);
    else if (static_cast<pos_type>(j) < m_indices.size())
      fn(m_indices.at(static_cast<pos_type>(j)), j);
    else {
      end = true;
      fn(1zu, j);
    }
  }

  void visitor::operator()(pos_type pos, const_func_ref fn) const {
    auto end = false;
    const auto signed_pos = static_cast<index_type>(pos);
    for (auto j = signed_pos - static_cast<index_type>(m_width); j < signed_pos; j++)
      visit(end, j, fn);
    visit(end, pos, fn);
    for (auto j = pos + 1; j <= pos + m_width; j++)
      visit(end, j, fn);
  }

  void visitor::operator()(pos_type pos, const std::function<void(pos_type)> &fn) const {
    (*this)(pos, [&fn](const auto i, const auto) { fn(i); });
  }

  skip_visitor::skip_visitor(const_indices_ref indices, unsigned long width)
    : visitor(indices, width) {
  }

  void skip_visitor::operator()(pos_type pos, const_func_ref fn) const {
    auto end = false;
    const auto signed_pos = static_cast<index_type>(pos);
    for (auto j = signed_pos - static_cast<index_type>(m_width); j < signed_pos; j++)
      visit(end, j, fn);
    for (auto j = pos + 1; j <= pos + m_width; j++)
      visit(end, static_cast<index_type>(j), fn);
  }
}
