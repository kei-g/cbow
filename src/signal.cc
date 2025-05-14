#include "signal.hh"

#include <fcntl.h>
#include <signal.h>
#include <sys/signalfd.h>
#include <unistd.h>

#include <cerrno>
#include <iostream>

signal::signal() {
  auto mask = sigset_t();
  sigemptyset(&mask);
  sigaddset(&mask, SIGINT);
  if (sigprocmask(SIG_BLOCK, &mask, nullptr) < 0) {
    std::cerr << std::format("sigprocmask: {}", strerror(errno)) << std::endl;
    throw;
  }
  auto fd = signalfd(-1, &mask, 0);
  if (fd < 0) {
    std::cerr << std::format("signalfd: {}", strerror(errno)) << std::endl;
    throw;
  }
  m_worker = std::thread([this](int fd) { poll(fd); }, fd);
}

bool signal::interrupted() {
  return m_interrupted.load(std::memory_order_acquire);
}

void signal::poll(int fd) {
  for (;;) {
    struct signalfd_siginfo info;
    auto rlen = read(fd, &info, sizeof(info));
    if (rlen < 0) {
      std::cerr << std::format("read([signalfd: {}]): {}", fd, strerror(errno)) << std::endl;
      throw;
    }
    if (static_cast<std::size_t>(rlen) < sizeof(info))
      continue;
    if (info.ssi_signo == SIGINT) {
      m_interrupted.store(true, std::memory_order_release);
      break;
    }
  }
  if (close(fd) < 0)
    std::cerr << std::format("close([signalfd: {}]): {}", fd, strerror(errno)) << std::endl;
}

signal::~signal() {
  if (interrupted())
    m_worker.join();
  else
    m_worker.detach();
}
