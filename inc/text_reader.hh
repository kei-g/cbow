#pragma once

#include <deque>
#include <istream>
#include <unordered_map>

#include "cbow/options.hh"
#include "cbow/trainer.hh"

struct text_reader {
private:
  std::deque<std::string> m_corpus;
  std::size_t m_count;
  std::deque<std::size_t> m_current_indices;
  std::deque<std::size_t> m_frequency;
  std::deque<std::deque<std::size_t>> m_indices;
  const cbow::options &m_options;
  std::unordered_map<std::string, std::size_t> m_reverse_lookup_table;

  void accept(const std::string &word);

  /**
   * Append an existing word
   */
  void append(const std::size_t index);

  /**
   * Append a word newly appeared
   */
  void append(const std::string &word);

  /**
   * Describe frequency of words in corpus.
   */
  void describe_frequency_of_words_to(std::ostream &dest) const;

public:
  text_reader() = delete;

  /**
   * Constructor
   */
  text_reader(const cbow::options &options);

  /**
   * Describe corpus and words.
   */
  void describe_to(std::ostream &dest) const;

  /**
   * Load from file descriptor.
   */
  void load(int fd);

  /**
   * Creates a CBoW model and trainer.
   */
  cbow::trainer populate(cbow::model *model, std::mt19937_64 &engine);
};
