// Times LU factorization (Gaussian elimination with partial pivoting) and one
// solve with the stored factors, for growing n. Prints: n, factor seconds,
// solve seconds. Compile: g++ -std=c++20 -O2 lu_timing.cpp -o lu_timing
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

using matrix = std::vector<std::vector<double>>;

// Overwrites a with L (below the diagonal, unit diagonal implied) and U.
// perm records the row swaps.
void lu_factor(matrix& a, std::vector<int>& perm) {
  const int n = a.size();
  for (int i = 0; i < n; i++) perm[i] = i;
  for (int k = 0; k < n; k++) {
    int p = k;                                   // partial pivoting
    for (int i = k + 1; i < n; i++)
      if (std::abs(a[i][k]) > std::abs(a[p][k])) p = i;
    std::swap(a[k], a[p]);
    std::swap(perm[k], perm[p]);
    for (int i = k + 1; i < n; i++) {
      double m = a[i][k] / a[k][k];              // the multiplier is l_ik
      a[i][k] = m;
      for (int j = k + 1; j < n; j++) a[i][j] -= m * a[k][j];
    }
  }
}

std::vector<double> lu_solve(const matrix& lu, const std::vector<int>& perm,
                             const std::vector<double>& b) {
  const int n = lu.size();
  std::vector<double> x(n);
  for (int i = 0; i < n; i++) {                  // forward: L y = P b
    double s = b[perm[i]];
    for (int j = 0; j < i; j++) s -= lu[i][j] * x[j];
    x[i] = s;
  }
  for (int i = n - 1; i >= 0; i--) {             // backward: U x = y
    double s = x[i];
    for (int j = i + 1; j < n; j++) s -= lu[i][j] * x[j];
    x[i] = s / lu[i][i];
  }
  return x;
}

int main() {
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int n : {100, 200, 400, 800, 1600, 3200}) {
    matrix a(n, std::vector<double>(n));
    for (auto& row : a) for (auto& x : row) x = u(rng);
    std::vector<double> b(n);
    for (auto& x : b) x = u(rng);
    matrix a0 = a;
    std::vector<int> perm(n);
    auto t0 = std::chrono::steady_clock::now();
    lu_factor(a, perm);
    auto t1 = std::chrono::steady_clock::now();
    const int reps = 20;
    std::vector<double> x;
    for (int r = 0; r < reps; r++) x = lu_solve(a, perm, b);
    auto t2 = std::chrono::steady_clock::now();
    double resid = 0;                            // check: max |A x - b|
    for (int i = 0; i < n; i++) {
      double s = -b[i];
      for (int j = 0; j < n; j++) s += a0[i][j] * x[j];
      resid = std::max(resid, std::abs(s));
    }
    std::printf("%d %.6e %.6e %.2e\n", n,
                std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t2 - t1).count() / reps, resid);
    std::fflush(stdout);
  }
}
