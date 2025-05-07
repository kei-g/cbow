#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace utf8 {
  struct reader;
}

struct file_descriptor {
  typedef std::function<void(const char *, uint32_t)> functor_type;
  typedef const functor_type &const_functor_ref;

private:
  int m_fd;
  std::size_t m_size;

  void close();

public:
  file_descriptor();
  file_descriptor(int fd);
  ~file_descriptor();

  bool open(const std::string &path);

  long read(void *dest, std::size_t len);

  void visit(const utf8::reader &cvt, const_functor_ref functor);
};
