// Times the tridiagonal (Thomas) solver for growing n, on the second-difference
// matrix of Lecture 12 with a known solution. Prints: n, seconds, max error.
// Compile: g++ -std=c++20 -O2 tridiagonal_timing.cpp -o tridiagonal_timing
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

// Solves A x = r for tridiagonal A with sub-diagonal a, diagonal b,
// super-diagonal c (a[0] and c[n-1] unused). No pivoting.
std::vector<double> solve_tridiagonal(const std::vector<double>& a,
                                      const std::vector<double>& b,
                                      const std::vector<double>& c,
                                      const std::vector<double>& r) {
  const int n = b.size();
  std::vector<double> u(n), y(n), x(n);
  u[0] = b[0];
  y[0] = r[0];
  for (int i = 1; i < n; i++) {        // factor and forward substitution
    double l = a[i] / u[i - 1];
    u[i] = b[i] - l * c[i - 1];
    y[i] = r[i] - l * y[i - 1];
  }
  x[n - 1] = y[n - 1] / u[n - 1];
  for (int i = n - 2; i >= 0; i--)     // back substitution
    x[i] = (y[i] - c[i] * x[i + 1]) / u[i];
  return x;
}

int main() {
  for (int n : {1000, 10000, 100000, 1000000, 10000000}) {
    std::vector<double> a(n, 1.0), b(n, -2.0), c(n, 1.0), x_true(n), r(n);
    for (int i = 0; i < n; i++) x_true[i] = std::sin(0.001 * i);
    for (int i = 0; i < n; i++) {      // r = A x_true
      r[i] = -2.0 * x_true[i];
      if (i > 0) r[i] += x_true[i - 1];
      if (i < n - 1) r[i] += x_true[i + 1];
    }
    int reps = n <= 100000 ? 50 : 3;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<double> x;
    for (int k = 0; k < reps; k++) x = solve_tridiagonal(a, b, c, r);
    auto t1 = std::chrono::steady_clock::now();
    double err = 0;
    for (int i = 0; i < n; i++) err = std::max(err, std::abs(x[i] - x_true[i]));
    std::printf("%d %.6e %.2e\n", n, std::chrono::duration<double>(t1 - t0).count() / reps, err);
  }
}
