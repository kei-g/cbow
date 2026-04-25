#pragma once

#include <deque>
#include <istream>
#include <unordered_map>

#include "cbow/options.hh"
#include "cbow/trainer.hh"

struct corpus_manager {
private:
  std::deque<std::deque<std::size_t>> m_corpus;
  std::deque<std::size_t> m_current_indices;
  std::deque<std::size_t> m_frequency;
  const cbow::options &m_options;
  std::unordered_map<std::string, std::size_t> m_reverse_lookup_table;
  std::size_t m_total_tokens;
  std::deque<std::string> m_vocabulary;

  void accept(const std::string &word);

  /**
   * Add token to corpus
   */
  void add_token_to_corpus(const std::size_t index);

  /**
   * Register a newly appeared word
   */
  void register_word_and_append(const std::string &word);

  /**
   * Describe frequency of vocabulary
   */
  void describe_frequency_of_vocabulary_to(std::ostream &dest) const;

public:
  corpus_manager() = delete;

  /**
   * Constructor
   */
  corpus_manager(const cbow::options &options);

  /**
   * Describe statistics of corpus and vocabulary.
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
