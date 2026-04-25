#include "utf8.hh"

static void call(const unsigned char buf[4], size_t &pos, uint32_t code, utf8::reader::const_functor_ref functor);
static bool process_1byte_utf8(const unsigned char buf[4], size_t &pos, utf8::reader::const_functor_ref functor);
static void process_2byte_utf8(const unsigned char buf[4], size_t &pos, utf8::reader::const_functor_ref functor);
static void process_3byte_utf8(const unsigned char buf[4], size_t &pos, utf8::reader::const_functor_ref functor);
static void process_4byte_utf8(const unsigned char buf[4], size_t &pos, utf8::reader::const_functor_ref functor);
static void throw_if_out_of_range(unsigned char c);
static void throw_if_out_of_range(unsigned char c0, unsigned char c1);

template <size_t N>
static bool enough(const unsigned char buf[4], const size_t pos) {
  static_assert(2 <= N && N <= 4);
  for (auto i = 1zu; i < N; i++) {
    if (pos < i + 1)
      return false;
    throw_if_out_of_range(buf[i]);
  }
  return true;
}

namespace utf8 {
  void reader::process(file_descriptor &fd, const_functor_ref functor) const {
    unsigned char buf[4];
    for (auto pos = 0zu;;) {
      auto rlen = fd.read(buf + pos, 1);
      if (rlen < 0)
        throw;
      if (rlen == 0) {
        if (pos)
          throw;
        break;
      }
      pos += rlen;
      const auto c0 = buf[0];
      if (!process_1byte_utf8(buf, pos, functor) && c0 < 0xe0)
        process_2byte_utf8(buf, pos, functor);
      else if (c0 < 0xf0)
        process_3byte_utf8(buf, pos, functor);
      else
        process_4byte_utf8(buf, pos, functor);
    }
  }
}

static void call(const unsigned char buf[4], size_t &pos, uint32_t code, utf8::reader::const_functor_ref functor) {
  char text[5];
  for (auto i = 0zu; i < pos; i++)
    text[i] = static_cast<char>(buf[i]);
  text[pos] = '\0';
  functor(text, code);
  pos = 0;
}

static bool process_1byte_utf8(const unsigned char buf[4], size_t &pos, utf8::reader::const_functor_ref functor) {
  const auto c0 = buf[0];
  if (c0 < 0x80) {
    call(buf, pos, c0, functor);
    return true;
  }
  return false;
}

static void process_2byte_utf8(const unsigned char buf[4], size_t &pos, utf8::reader::const_functor_ref functor) {
  const auto c0 = buf[0];
  if (c0 < 0xc2)
    throw;
  if (enough<2>(buf, pos)) {
    const auto c1 = buf[1];
    call(buf, pos, ((c0 & 31) << 6) | (c1 & 63), functor);
  }
}

static void process_3byte_utf8(const unsigned char buf[4], size_t &pos, utf8::reader::const_functor_ref functor) {
  if (enough<3>(buf, pos)) {
    const auto c0 = buf[0];
    const auto c1 = buf[1];
    throw_if_out_of_range(c0, c1);
    const auto c2 = buf[2];
    call(buf, pos, ((c0 & 15) << 12) | ((c1 & 63) << 6) | (c2 & 63), functor);
  }
}

static void process_4byte_utf8(const unsigned char buf[4], size_t &pos, utf8::reader::const_functor_ref functor) {
  const auto c0 = buf[0];
  if (0xf4 < c0)
    throw;
  if (enough<4>(buf, pos)) {
    const auto c1 = buf[1];
    const auto c2 = buf[2];
    throw_if_out_of_range(c0, c1);
    const auto c3 = buf[3];
    call(buf, pos, ((c0 & 7) << 18) | ((c1 & 63) << 12) | ((c2 & 63) << 6) | (c3 & 63), functor);
  }
}

static void throw_if_out_of_range(unsigned char c) {
  if (c < 0x80 || 0xbf < c)
    throw;
}

static void throw_if_out_of_range(unsigned char c0, unsigned char c1) {
  if (c0 == 0xe0 && 0x80 <= c1 && c1 < 0xa0)
    throw;
  if (c0 == 0xf0 && 0x80 <= c1 && c1 < 0x90)
    throw;
  if (c0 == 0xed && 0xa0 <= c1)
    throw;
  if (c0 == 0xf4 && 0x90 <= c1)
    throw;
}
