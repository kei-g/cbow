#include "corpus_manager.hh"

#include <algorithm>
#include <iostream>
#include <regex>
#include <sstream>

#include "cbow/trainer.hh"

using namespace std;

template <typename T>
concept HasSize = requires(const T &target) { target.size(); };

/**
 * Zip elements in `arg2` and `arg1` to the specified container, and return the largest size of element in `arg1`
 */
template <HasSize T, typename U>
static size_t zip_to(deque<pair<U, T>> *dest, const deque<T> &arg1, const deque<U> &arg2) {
  auto largest = numeric_limits<size_t>::min();
  for (auto i = 0zu; i < arg1.size(); i++) {
    const auto &elem = arg1.at(i);
    dest->push_back(make_pair(arg2.at(i), elem));
    const auto size = elem.size();
    if (largest < size)
      largest = size;
  }
  return largest;
}

corpus_manager::corpus_manager(const cbow::options &options)
  : m_options(options)
  , m_total_tokens(0zu) {
  m_frequency.push_back(0zu);
  m_frequency.push_back(0zu);
  m_reverse_lookup_table.insert(make_pair("<|beginoftext|>", 0zu));
  m_reverse_lookup_table.insert(make_pair("<|endoftext|>", 1zu));
  m_vocabulary.push_back("<|beginoftext|>");
  m_vocabulary.push_back("<|endoftext|>");
}

void corpus_manager::accept(const string &word) {
  const auto found = m_reverse_lookup_table.find(word);
  if (found == m_reverse_lookup_table.cend())
    register_word_and_append(word);
  else
    add_token_to_corpus(found->second);
}

void corpus_manager::add_token_to_corpus(const size_t index) {
  if (m_options.verbosity & 4)
    cerr << format("\x1b[35m{}\x1b[m = {}", m_vocabulary.at(index), index) << endl;
  if (index < 2zu) {
    m_corpus.push_back(std::move(m_current_indices));
    m_current_indices = std::deque<std::size_t>();
  }
  else {
    m_current_indices.push_back(index);
    m_frequency[index]++;
    m_total_tokens++;
  }
}

void corpus_manager::register_word_and_append(const string &word) {
  const auto index = m_reverse_lookup_table.size();
  if (m_options.verbosity & 8)
    cerr << format("\x1b[32m{}\x1b[m = {}", word, index) << endl;
  m_current_indices.push_back(index);
  m_frequency.push_back(1zu);
  m_reverse_lookup_table.insert(make_pair(word, index));
  m_total_tokens++;
  m_vocabulary.push_back(word);
}

void corpus_manager::describe_frequency_of_vocabulary_to(ostream &dest) const {
  auto vocabulary_with_freq = deque<pair<size_t, string>>();
  const auto longest = zip_to(&vocabulary_with_freq, m_vocabulary, m_frequency);

  // remove <|beginoftext|> <|endoftext|>
  vocabulary_with_freq.pop_front();
  vocabulary_with_freq.pop_front();

  // header line
  dest << format("{0:{1}} 頻度 割合", "単語", longest) << endl;

  // sort vocabulary by frequency in descending order
  sort(vocabulary_with_freq.begin(), vocabulary_with_freq.end(),
       [](const auto &lhs, const auto &rhs) { return rhs.first < lhs.first; });

  // the average of frequency depends on the ratio of word count and vocabulary size
  const auto average = static_cast<double>(m_total_tokens) / vocabulary_with_freq.size();

  // the variance of frequency, be accumulated during iteration of words in vocabulary,
  // and be divided by its size finally
  auto variance = static_cast<double>(0);

  // show frequency of words in vocabulary, accumulate squares of difference from average simultaneously
  for (const auto &[freq, word] : vocabulary_with_freq) {
    const auto percent = static_cast<float>(freq) * 100 / m_total_tokens;
    dest << format("{0:{1}} {2:>4} {3}%", word, longest, freq, percent) << endl;
    const auto difference = freq - average;
    variance += difference * difference;
  }

  // correct `variance` here
  variance /= vocabulary_with_freq.size();

  // standard deviation
  const auto stddev = sqrt(variance);

  // show average, variance, stddev
  dest << format("頻度: {{平均: {}, 分散: {}, 標準偏差: {}}}", average, variance, stddev) << endl;
}

void corpus_manager::describe_to(ostream &dest) const {
  const auto &o = m_options;

  if (o.verbosity & 2)
    describe_frequency_of_vocabulary_to(dest);

  if (o.verbosity & 1) {
    dest << format("総単語数: {}語, 語彙: {}語", m_total_tokens, m_vocabulary.size() - 1) << endl;
    dest << format("周囲{}単語を用いて{}次元のベクトルへ変換します", o.width, o.dimensions) << endl;
  }
}

cbow::trainer corpus_manager::populate(cbow::model *model, std::mt19937_64 &engine) {
  const auto H = m_options.dimensions;
  const auto V = m_vocabulary.size();
  model->in_matrix = std::make_unique<cbow::matrix_type>(V, H, engine);
  model->out_matrix = std::make_unique<cbow::matrix_type>(H, V, engine);
  model->vocabulary = std::make_unique<std::vector<std::string>>(std::move_iterator(m_vocabulary.begin()),
                                                                 std::move_iterator(m_vocabulary.end()));
  return cbow::trainer(engine, m_corpus, *model, m_options);
}

#include "tokenizer.hh"
#include "utf8.hh"

void corpus_manager::load(int fd) {
  auto file = file_descriptor(fd);
  auto reader = utf8::reader();
  auto lower = string();
  auto receiver = [this, &lower](const string &word) {
    if (m_options.case_sensitive)
      accept(word);
    else {
      lower.clear();
      transform(word.begin(), word.end(), back_inserter(lower), [](const auto c) { return std::tolower(c); });
      accept(lower);
    }
  };
  auto tokenize = tokenizer(receiver);
  file.visit(reader, tokenize);
}
