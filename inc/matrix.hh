#pragma once

#include <cmath>
#include <concepts>
#include <memory>
#include <ostream>
#include <random>
#ifdef USE_SPAN
  #include <span>
#endif /* USE_SPAN */
#include <vector>

template <std::floating_point T>
struct matrix_row;

template <std::floating_point T>
struct matrix {
  friend std::ostream &operator<<(std::ostream &os, const matrix<T> &mat);
  friend struct matrix_row<T>;

private:
  std::vector<T> elements;

public:
  const std::size_t columns;
  const std::size_t rows;

public:
  matrix() = delete;
  matrix(const matrix<T> &) = delete;
  matrix(matrix<T> &&) = delete;

  matrix(const std::size_t columns, const std::size_t rows, std::mt19937_64 &engine)
    : elements(columns * rows)
    , columns(columns)
    , rows(rows) {
    const auto N = static_cast<std::size_t>(columns * rows);
    auto dist = std::normal_distribution<T>(0, 1 / std::sqrt(N));
    for (auto i = 0zu; i < N;) {
      elements[i++] = dist(engine);
      if (i == N)
        break;
      elements[i++] = dist(engine);
    }
  }

  matrix(const std::size_t columns, const std::size_t rows, const std::vector<T> &elements)
    : elements(elements)
    , columns(columns)
    , rows(rows) {
  }

  matrix(const matrix<T> *source)
    : elements(source->columns * source->rows)
    , columns(source->rows)
    , rows(source->columns) {
    for (auto c = 0zu, i = 0zu; i < rows; i++)
      for (auto j = 0zu; j < columns; j++) {
        const auto x = j * rows + i;
        if (source->elements.size() <= x)
          throw;
        elements[c++] = source->elements[x];
      }
  }

#ifdef USE_SPAN
  auto operator[](const auto row) {
    if (row < 0 || rows <= row)
      throw;
    return std::span(> elements).subspan(columns * row, columns);
  }
#endif /* USE_SPAN */

  auto operator*(const std::vector<T> &vec) const {
    auto result = std::vector<T>(rows);
    for (auto i = 0zu, j = 0zu; i < rows; i++) {
      auto work = static_cast<T>(0);
      for (const auto value : vec)
        work += elements[j++] * value;
      result[i] = work;
    }
    return result;
  }

  void copy_to(std::unique_ptr<matrix<T>> &dest) const {
    dest = std::make_unique<matrix<T>>(columns, rows, elements);
  }

  auto row(const auto row) const {
    if (row < 0 || rows <= row)
      throw;
    return matrix_row<T>(this, row);
  }

  auto transpose(void) const {
    return std::make_unique<matrix<T>>(this);
  }

  void update(auto eta, const auto &inference) {
    auto c = 0zu;
    for (auto i = 0zu; i < this->rows; i++) {
      const auto coeff = eta * inference.probability.at(i);
      for (auto j = 0zu; j < this->columns; j++)
        elements[c++] -= coeff * inference.hidden.at(j);
    }
  }

  void update(auto eta, const auto &hadamard, const auto &indices, const auto pos, const auto width) {
    const auto coeff = eta / (width * 2);
    for (auto i = 0zu, offset = 0zu; i < rows; i++, offset += columns) {
      const auto value = coeff * hadamard.at(i);
#ifdef USE_SPAN
      auto row = (*this)[i];
      for (auto j = pos - width; j + 1 < pos; j++)
        row[indices.at(j)] -= value;
      row[indices.at(pos)] -= value;
      for (auto j = pos + 1; j <= pos + width; j++)
        row[indices.at(j)] -= value;
#else  /* USE_SPAN */
      for (auto j = pos - width; j + 1 < pos; j++)
        elements[offset + indices.at(j)] -= value;
      elements[offset + indices.at(pos)] -= value;
      for (auto j = pos + 1; j <= pos + width; j++)
        elements[offset + indices.at(j)] -= value;
#endif /* USE_SPAN */
    }
  }
};

template <std::floating_point T>
struct matrix_row {
  friend struct matrix<T>;

private:
  const std::vector<T> &elements;
  const std::size_t offset;

public:
  const std::size_t columns;

private:
  matrix_row(const matrix<T> *mat, const std::size_t row)
    : elements(mat->elements)
    , offset(mat->columns * row)
    , columns(mat->columns) {
  }

public:
  matrix_row() = delete;
  matrix_row(const matrix_row &) = delete;
  matrix_row(matrix_row &&) = delete;

  const T &operator[](auto col) const {
    if (col < 0 || columns <= col)
      throw;
    return elements.at(col + offset);
  }
};

#include <format>

template <std::floating_point T>
std::ostream &operator<<(std::ostream &os, const matrix<T> &mat) {
  os << '[' << std::endl;
  const auto c = mat.columns;
  const auto &e = mat.elements;
  const auto r = mat.rows;
  for (auto i = 0zu, t = 0zu; i < r; i++) {
    os << "  [";
    for (auto j = 0zu; j < c; j++) {
      os << std::format("    {}", e.at(t++));
      if (j + 1 < c)
        os << ',';
      os << std::endl;
    }
    os << "]";
    if (i + 1 < r)
      os << ',';
    os << std::endl;
  }
  os << ']' << std::endl;
  return os;
}
