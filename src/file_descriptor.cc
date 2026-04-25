#include "file_descriptor.hh"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <iostream>

#include "utf8.hh"

file_descriptor::file_descriptor()
  : m_fd(-1) {
}

file_descriptor::file_descriptor(int fd)
  : m_fd(fd) {
  if (fd < 0)
    throw;
}

file_descriptor::~file_descriptor() {
  close();
}

void file_descriptor::close() {
  if (0 < m_fd && ::close(m_fd) < 0)
    perror("close");
  m_fd = -1;
}

/**
 * XXX: Not in use now, however, `m_size` will be used to show progress during reading.
 */
bool file_descriptor::open(const std::string &path) {
  close();
  m_fd = ::open(path.c_str(), O_RDONLY);
  if (m_fd < 0) {
    std::cerr << std::format("open(\"{}\"): {}", path, strerror(errno)) << std::endl;
    return false;
  }
  const auto size = lseek(m_fd, 0, SEEK_END);
  if (size < 0) {
    std::cerr << std::format("lseek(\"{}\", 0, SEEK_END): {}", path, strerror(errno)) << std::endl;
    close();
    return false;
  }
  m_size = size;
  if (lseek(m_fd, 0, SEEK_SET) < 0) {
    std::cerr << std::format("lseek(\"{}\", 0, SEEK_SET): {}", path, strerror(errno)) << std::endl;
    close();
    return false;
  }
  return true;
}

long file_descriptor::read(void *dest, std::size_t len) {
  return ::read(m_fd, dest, len);
}

void file_descriptor::visit(const utf8::reader &u8r, const_functor_ref functor) {
  u8r.process(*this, functor);
}
