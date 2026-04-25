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

/**
 * @brief A matrix implementation optimized for neural network weights.
 *
 * This class stores matrix elements in a flat std::vector for cache efficiency.
 * It provides specialized methods for the CBOW architecture,
 * including matrix-vector products for the forward pass and custom update rules for stochastic gradient descent.
 *
 * @tparam T Floating-point type (float, double) for calculations.
 */
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

  /**
   * @brief Constructs a matrix with elements initialized from a normal distribution with mean 0 and stddev 1/sqrt(N)
   * (Xavier-like initialization).
   *
   * @param columns Number of columns.
   * @param rows Number of rows.
   * @param engine Random number generator.
   */
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

  /**
   * @brief Constructs a matrix by copying elements from the provided vector.
   *
   * @param columns Number of columns.
   * @param rows Number of rows.
   * @param elements Vector of elements.
   */
  matrix(const std::size_t columns, const std::size_t rows, const std::vector<T> &elements)
    : elements(elements)
    , columns(columns)
    , rows(rows) {
  }

  /**
   * @brief Constructs a transposed matrix from the specified source matrix.
   *
   * @param source Source matrix.
   */
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

  /**
   * @brief Performs matrix-vector multiplication,
   * typically used to compute output layer scores from the hidden layer.
   *
   * @param vec Vector of values.
   * @return Result of matrix-vector multiplication.
   */
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

  /**
   * @brief Computes a weighted sum of matrix rows,
   * used for backpropagating errors to the embedding layer.
   *
   * @param vec Vector of values.
   * @return Weighted sum of matrix rows.
   */
  auto multiply_transpose(const std::vector<T> &vec) const {
    auto result = std::vector<T>(columns);
    for (auto c = 0zu, i = 0zu; i < rows; i++) {
      const auto value = vec.at(i);
      for (auto j = 0zu; j < columns; j++)
        result[j] += elements[c++] * value;
    }
    return result;
  }

  auto row(const auto row) const {
    if (row < 0 || rows <= row)
      throw;
    return matrix_row<T>(this, row);
  }

  auto transpose(void) const {
    return std::make_unique<matrix<T>>(this);
  }

  /**
   * @brief Updates weights using the inference results (probability and hidden state).
   * Used for the output matrix backpropagation.
   *
   * @param eta Learning rate.
   * @param inference Inference results.
   */
  void update(auto eta, const auto &inference) {
    auto c = 0zu;
    for (auto i = 0zu; i < this->rows; i++) {
      const auto coeff = eta * inference.probability.at(i);
      for (auto j = 0zu; j < this->columns; j++)
        elements[c++] -= coeff * inference.hidden.at(j);
    }
  }

  /**
   * @brief Updates weights by propagating error gradients from the output layer to the embedding matrix.
   *
   * @param eta Learning rate.
   * @param hidden_gradient Weighted error gradient.
   * @param pos Position index.
   * @param visit Callback function to access embedding matrix indices.
   * @param width Window size.
   */
  void update(auto eta, const auto &hidden_gradient, const auto pos, const auto &visit, const auto width) {
    const auto coeff = eta / (width * 2);
    for (auto i = 0zu, offset = 0zu; i < rows; i++, offset += columns) {
      const auto value = coeff * hidden_gradient.at(i);
#ifdef USE_SPAN
      auto row = (*this)[i];
      for (auto j = pos - width; j + 1 < pos; j++)
        row[indices.at(j)] -= value;
      row[indices.at(pos)] -= value;
      for (auto j = pos + 1; j <= pos + width; j++)
        row[indices.at(j)] -= value;
#else  /* USE_SPAN */
      visit(pos, [this, offset, value](const auto i) { elements[offset + i] -= value; });
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
