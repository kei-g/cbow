#pragma once

#include <cctype>
#include <cstdint>
#include <string>

template <typename T>
concept TextAcceptable = requires(T &target, const std::string &word) { target(word); };

bool is_ucs4_space(uint32_t code);

template <TextAcceptable T>
struct tokenizer {
private:
  T &m_receiver;
  std::string m_text;

  bool is_in_progress_for_endoftext(const char u8char) const {
    return (m_text.empty() && u8char == '<') || ((m_text == "<" || m_text == "<|endoftext") && u8char == '|');
  }

  void prune_incomplete_endoftext() {
    if (m_text.starts_with('<')) {
      m_receiver("<");
      m_text = m_text.substr(1);
      if (m_text.starts_with('|')) {
        m_receiver("|");
        m_text = m_text.substr(1);
        if (m_text.ends_with('|')) {
          m_text.pop_back();
          m_receiver(m_text);
          m_text.erase();
          m_receiver("|");
        }
      }
    }
  }

public:
  tokenizer(T &receiver)
    : m_receiver(receiver) {
  }

  void operator()(const char *u8char, uint32_t code) {
    if (std::isalnum(*u8char) || is_in_progress_for_endoftext(*u8char))
      m_text += *u8char;
    else if (m_text == "<|endoftext|" && *u8char == '>') {
      m_text += *u8char;
      m_receiver(m_text);
      m_text.erase();
    }
    else {
      prune_incomplete_endoftext();
      if (!m_text.empty()) {
        m_receiver(m_text);
        m_text.erase();
      }
      if (!is_ucs4_space(code))
        m_receiver(u8char);
    }
  }
};
