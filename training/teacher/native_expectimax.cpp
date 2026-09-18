#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

// Heuristic constants and expectimax search shape are adapted from the MIT-licensed
// nneonneo/2048-ai implementation: https://github.com/nneonneo/2048-ai

namespace {

constexpr std::uint64_t kRowMask = 0xFFFF;
constexpr int kBoardCells = 16;
constexpr int kCellBits = 4;
constexpr int kRowBits = 16;
constexpr double kProbabilityThreshold = 0.0001;

struct Tables {
  std::array<std::uint16_t, 1 << kRowBits> left{};
  std::array<std::uint16_t, 1 << kRowBits> right{};
  std::array<double, 1 << kRowBits> heuristic{};

  static std::uint16_t reverse_row(std::uint16_t row) {
    return static_cast<std::uint16_t>(((row & 0x000F) << 12) |
                                      ((row & 0x00F0) << 4) |
                                      ((row & 0x0F00) >> 4) |
                                      ((row & 0xF000) >> 12));
  }

  Tables() {
    for (std::uint32_t encoded = 0; encoded < left.size(); ++encoded) {
      std::array<int, 4> values{};
      std::vector<int> tiles;
      for (int index = 0; index < 4; ++index) {
        values[index] = (encoded >> (kCellBits * index)) & 0xF;
        if (values[index] != 0) {
          tiles.push_back(values[index]);
        }
      }

      std::vector<int> merged;
      for (std::size_t source = 0; source < tiles.size();) {
        int value = tiles[source];
        if (source + 1 < tiles.size() && tiles[source + 1] == value) {
          ++value;
          source += 2;
        } else {
          ++source;
        }
        merged.push_back(value);
      }
      std::uint16_t moved = 0;
      for (std::size_t index = 0; index < merged.size(); ++index) {
        moved |= static_cast<std::uint16_t>(merged[index] << (kCellBits * index));
      }
      left[encoded] = moved;

      double rank_sum = 0.0;
      int empty = 0;
      int merges = 0;
      int previous = 0;
      int counter = 0;
      for (int value : values) {
        rank_sum += std::pow(static_cast<double>(value), 3.5);
        if (value == 0) {
          ++empty;
          continue;
        }
        if (previous == value) {
          ++counter;
        } else if (counter > 0) {
          merges += 1 + counter;
          counter = 0;
        }
        previous = value;
      }
      if (counter > 0) {
        merges += 1 + counter;
      }

      double monotonicity_left = 0.0;
      double monotonicity_right = 0.0;
      for (int index = 0; index < 3; ++index) {
        const double current = std::pow(static_cast<double>(values[index]), 4.0);
        const double following = std::pow(static_cast<double>(values[index + 1]), 4.0);
        if (values[index] > values[index + 1]) {
          monotonicity_left += current - following;
        } else {
          monotonicity_right += following - current;
        }
      }
      heuristic[encoded] = 200000.0 + 270.0 * empty + 700.0 * merges -
                           47.0 * std::min(monotonicity_left, monotonicity_right) -
                           11.0 * rank_sum;
    }

    for (std::uint32_t encoded = 0; encoded < right.size(); ++encoded) {
      right[encoded] = reverse_row(left[reverse_row(static_cast<std::uint16_t>(encoded))]);
    }
  }
};

const Tables& tables() {
  static const Tables value;
  return value;
}

std::uint64_t transpose(std::uint64_t board) {
  std::uint64_t result = 0;
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      const int source = kCellBits * (row * 4 + column);
      const int target = kCellBits * (column * 4 + row);
      result |= ((board >> source) & 0xF) << target;
    }
  }
  return result;
}

std::uint64_t slide_rows(std::uint64_t board, bool right) {
  const auto& moves = right ? tables().right : tables().left;
  std::uint64_t result = 0;
  for (int row = 0; row < 4; ++row) {
    const auto encoded = static_cast<std::uint16_t>((board >> (kRowBits * row)) & kRowMask);
    result |= static_cast<std::uint64_t>(moves[encoded]) << (kRowBits * row);
  }
  return result;
}

std::uint64_t execute_move(std::uint64_t board, int direction) {
  if (direction == 3) {
    return slide_rows(board, false);
  }
  if (direction == 1) {
    return slide_rows(board, true);
  }
  const std::uint64_t transposed = transpose(board);
  return transpose(slide_rows(transposed, direction == 2));
}

double score_heuristic(std::uint64_t board) {
  double score = 0.0;
  const std::uint64_t transposed = transpose(board);
  for (int index = 0; index < 4; ++index) {
    score += tables().heuristic[(board >> (kRowBits * index)) & kRowMask];
    score += tables().heuristic[(transposed >> (kRowBits * index)) & kRowMask];
  }
  return score;
}

using TranspositionTable =
    std::unordered_map<std::uint64_t, std::pair<int, double>>;

double score_tilechoose_node(std::uint64_t board, double cumulative_probability,
                             int depth, int depth_limit,
                             TranspositionTable& transposition_table);

double score_move_node(std::uint64_t board, double cumulative_probability, int depth,
                       int depth_limit, TranspositionTable& transposition_table) {
  double best = 0.0;
  for (int direction = 0; direction < 4; ++direction) {
    const std::uint64_t moved = execute_move(board, direction);
    if (moved == board) {
      continue;
    }
    best = std::max(best, score_tilechoose_node(moved, cumulative_probability,
                                                depth + 1, depth_limit,
                                                transposition_table));
  }
  return best;
}

double score_tilechoose_node(std::uint64_t board, double cumulative_probability,
                             int depth, int depth_limit,
                             TranspositionTable& transposition_table) {
  if (cumulative_probability < kProbabilityThreshold || depth >= depth_limit) {
    return score_heuristic(board);
  }

  if (depth < 15) {
    const auto cached = transposition_table.find(board);
    if (cached != transposition_table.end() && cached->second.first <= depth) {
      return cached->second.second;
    }
  }

  std::array<int, kBoardCells> empty_shifts{};
  int empty_count = 0;
  for (int index = 0; index < kBoardCells; ++index) {
    if (((board >> (kCellBits * index)) & 0xF) == 0) {
      empty_shifts[empty_count++] = kCellBits * index;
    }
  }
  if (empty_count == 0) {
    return score_move_node(board, cumulative_probability, depth, depth_limit,
                           transposition_table);
  }

  const double cell_probability = 1.0 / empty_count;
  double expected = 0.0;
  for (int index = 0; index < empty_count; ++index) {
    const int shift = empty_shifts[index];
    expected += cell_probability * 0.9 *
                score_move_node(board | (std::uint64_t{1} << shift),
                                cumulative_probability * cell_probability * 0.9,
                                depth, depth_limit, transposition_table);
    expected += cell_probability * 0.1 *
                score_move_node(board | (std::uint64_t{2} << shift),
                                cumulative_probability * cell_probability * 0.1,
                                depth, depth_limit, transposition_table);
  }

  if (depth < 15) {
    transposition_table[board] = {depth, expected};
  }
  return expected;
}

}  // namespace

extern "C" int expectimax_policy(std::uint64_t board, int minimum_depth,
                                 double temperature, float* output) {
  const int depth_limit = minimum_depth;
  std::array<double, 4> scores{};
  std::array<bool, 4> legal{};
  int legal_count = 0;
  double maximum = -std::numeric_limits<double>::infinity();

  for (int direction = 0; direction < 4; ++direction) {
    const std::uint64_t moved = execute_move(board, direction);
    if (moved == board) {
      continue;
    }
    TranspositionTable transposition_table;
    transposition_table.reserve(32768);
    scores[direction] =
        score_tilechoose_node(moved, 1.0, 0, depth_limit, transposition_table);
    legal[direction] = true;
    maximum = std::max(maximum, scores[direction]);
    ++legal_count;
  }

  double total = 0.0;
  for (int direction = 0; direction < 4; ++direction) {
    if (!legal[direction]) {
      output[direction] = 0.0F;
      continue;
    }
    const double centered =
        std::max((scores[direction] - maximum) / temperature, -80.0);
    output[direction] = static_cast<float>(std::exp(centered));
    total += output[direction];
  }
  if (total > 0.0) {
    for (int direction = 0; direction < 4; ++direction) {
      output[direction] = static_cast<float>(output[direction] / total);
    }
  }
  return legal_count;
}
