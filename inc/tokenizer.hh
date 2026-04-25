#pragma once

#include <cctype>
#include <cstdint>
#include <string>

template <typename T>
concept TextAcceptable = requires(T &target, const std::string &word) { target(word); };

bool is_utf32_space(uint32_t code);

/**
 * @brief A UTF-8 aware tokenizer that extracts words and recognizes special tokens like "<|endoftext|>"
 *
 * @tparam T The type of the receiver that accepts extracted words.
 */
template <TextAcceptable T>
struct tokenizer {
private:
  std::string m_current_token;
  T &m_receiver;

  bool is_in_progress_for_endoftext(const char u8char) const {
    return (m_current_token.empty() && u8char == '<') ||
           ((m_current_token == "<" || m_current_token == "<|endoftext") && u8char == '|');
  }

  void flush_partial_token() {
    if (m_current_token.starts_with('<')) {
      m_receiver("<");
      m_current_token = m_current_token.substr(1);
      if (m_current_token.starts_with('|')) {
        m_receiver("|");
        m_current_token = m_current_token.substr(1);
        if (m_current_token.ends_with('|')) {
          m_current_token.pop_back();
          m_receiver(m_current_token);
          m_current_token.erase();
          m_receiver("|");
        }
      }
    }
  }

public:
  tokenizer(T &receiver)
    : m_receiver(receiver) {
  }

  /**
   * @brief Processes a single character and its Unicode code point,
   * accumulating characters into words and notifying the receiver.
   *
   * @param u8char The UTF-8 encoded character.
   * @param code The Unicode code point.
   */
  void operator()(const char *utf8_char, uint32_t code) {
    if (std::isalnum(*utf8_char) || is_in_progress_for_endoftext(*utf8_char))
      m_current_token += *utf8_char;
    else if (m_current_token == "<|endoftext|" && *utf8_char == '>') {
      m_current_token += *utf8_char;
      m_receiver(m_current_token);
      m_current_token.erase();
    }
    else {
      flush_partial_token();
      if (!m_current_token.empty()) {
        m_receiver(m_current_token);
        m_current_token.erase();
      }
      if (!is_utf32_space(code))
        m_receiver(utf8_char);
    }
  }
};
