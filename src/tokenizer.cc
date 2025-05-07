#include "tokenizer.hh"

bool is_ucs4_space(uint32_t code) {
  switch (code) {
  case ' ':
  case '\n':
  case '\r':
  case '\t':
  case 0xa0:
  case 0x1680:
  case 0x200b:
  case 0x202f:
  case 0x205f:
  case 0x2060:
  case 0x3000:
    return true;
  }
  return 0x2000 <= code && code <= 0x200a;
}
