#pragma once

#include <thread>

struct signal {
private:
  std::atomic<bool> m_interrupted;
  std::thread m_worker;

  void poll(int fd);

public:
  signal();
  ~signal();

  bool interrupted();
};
