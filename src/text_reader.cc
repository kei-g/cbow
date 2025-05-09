#include "text_reader.hh"

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

text_reader::text_reader(const cbow::options &options)
  : m_count(0zu)
  , m_options(options) {
  m_corpus.push_back("<|beginoftext|>");
  m_corpus.push_back("<|endoftext|>");
  m_frequency.push_back(0zu);
  m_frequency.push_back(0zu);
  m_reverse_lookup_table.insert(make_pair("<|beginoftext|>", 0zu));
  m_reverse_lookup_table.insert(make_pair("<|endoftext|>", 1zu));
}

void text_reader::accept(const string &word) {
  const auto found = m_reverse_lookup_table.find(word);
  if (found == m_reverse_lookup_table.cend())
    append(word);
  else
    append(found->second);
}

void text_reader::append(const size_t index) {
  if (m_options.verbosity & 4)
    cerr << format("\x1b[35m{}\x1b[m = {}", m_corpus.at(index), index) << endl;
  if (index < 2zu) {
    m_indices.push_back(std::move(m_current_indices));
    m_current_indices = std::deque<std::size_t>();
  }
  else {
    m_count++;
    m_frequency[index]++;
    m_current_indices.push_back(index);
  }
}

void text_reader::append(const string &word) {
  const auto index = m_reverse_lookup_table.size();
  if (m_options.verbosity & 8)
    cerr << format("\x1b[32m{}\x1b[m = {}", word, index) << endl;
  m_corpus.push_back(word);
  m_count++;
  m_current_indices.push_back(index);
  m_frequency.push_back(1zu);
  m_reverse_lookup_table.insert(make_pair(word, index));
}

void text_reader::describe_frequency_of_words_to(ostream &dest) const {
  auto corpus = deque<pair<size_t, string>>();
  const auto longest = zip_to(&corpus, m_corpus, m_frequency);

  // remove <|beginoftext|> <|endoftext|>
  corpus.pop_front();
  corpus.pop_front();

  // header line
  dest << format("{0:{1}} 頻度 割合", "単語", longest) << endl;

  // sort corpus by frequency in descending order
  sort(corpus.begin(), corpus.end(), [](const auto &lhs, const auto &rhs) { return rhs.first < lhs.first; });

  // the average of frequency depends on the ratio of word count and corpus size
  const auto average = static_cast<double>(m_count) / corpus.size();

  // the variance of frequency, be accumulated during iteration of words in corpus, and be divided by its size finally
  auto variance = static_cast<double>(0);

  // show frequency of words, accumulate squares of difference from average simultaneously
  for (const auto &[freq, word] : corpus) {
    const auto percent = static_cast<float>(freq) * 100 / m_count;
    dest << format("{0:{1}} {2:>4} {3}%", word, longest, freq, percent) << endl;
    const auto difference = freq - average;
    variance += difference * difference;
  }

  // correct `variance` here
  variance /= corpus.size();

  // standard deviation
  const auto stddev = sqrt(variance);

  // show average, variance, stddev
  dest << format("頻度: {{平均: {}, 分散: {}, 標準偏差: {}}}", average, variance, stddev) << endl;
}

void text_reader::describe_to(ostream &dest) const {
  const auto &o = m_options;

  if (o.verbosity & 2)
    describe_frequency_of_words_to(dest);

  if (o.verbosity & 1) {
    dest << format("総単語数: {}語, 語彙: {}語", m_count, m_corpus.size() - 1) << endl;
    dest << format("周囲{}単語を用いて{}次元のベクトルへ変換します", o.width, o.dimensions) << endl;
  }
}

cbow::trainer text_reader::populate(cbow::model *model, std::mt19937_64 &engine) {
  const auto H = m_options.dimensions;
  const auto V = m_corpus.size();
  model->corpus = std::make_unique<std::vector<std::string>>(std::move_iterator(m_corpus.begin()),
                                                             std::move_iterator(m_corpus.end()));
  model->in_matrix = std::make_unique<cbow::matrix_type>(V, H, engine);
  model->out_matrix = std::make_unique<cbow::matrix_type>(H, V, engine);
  return cbow::trainer(m_indices, m_options);
}

#include "tokenizer.hh"
#include "utf8.hh"

void text_reader::load(int fd) {
  auto file = file_descriptor(fd);
  auto reader = utf8::reader();
  auto receiver = [this](const string &word) {
    if (m_options.case_sensitive)
      accept(word);
    else {
      auto lower = string();
      transform(word.begin(), word.end(), back_inserter(lower), [](const auto c) { return std::tolower(c); });
      accept(lower);
    }
  };
  auto tok = tokenizer(receiver);
  file.visit(reader, tok);
}
