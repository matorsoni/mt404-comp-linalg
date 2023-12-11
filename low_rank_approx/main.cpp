#include <iostream>
#include <vector>

#include "low_rank_approx.hpp"

using namespace std;
using namespace Eigen;

bool test_prime_factors(int n);
bool test_svd(const MatrixXd& mat);

int main() {
    vector<MatrixXd> test_matrices;
    test_matrices.emplace_back(MatrixXd::Random(2, 2));
    test_matrices.emplace_back(MatrixXd::Random(6, 4));
    test_matrices.emplace_back(MatrixXd::Random(8, 9));
    test_matrices.emplace_back(MatrixXd::Random(14, 24));

    MatrixXd matrix(3, 2);
    matrix << 1, 2,
              3, 4,
              5, 6;

    // Display the original matrix, U, Sigma, and V
    auto [U, sigma, V] = SVD_eigen(matrix);
    cout << "Original Matrix:\n" << matrix << "\n\n";
    cout << "U Matrix:\n" << U << "\n\n";
    cout << "Singular Values (Sigma):\n" << sigma << "\n\n";
    cout << "V Matrix:\n" << V << "\n";

    cout << U.col(0).norm() << "\n";

    MatrixXd C = MatrixXd::Random(6, 4);
    const auto [A,B] = low_rank_approx(C, 3, 2, 2, 2);
    const auto K = kroneckerProduct(A, B);
    cout << "Original Matrix:\n" << C << "\n\n";
    cout << "A:\n" << A << "\n\n";
    cout << "B:\n" << B << "\n\n";
    cout << "AxB:\n" << K << "\n\n";

    cout << spectral_norm(C-K) << "\n";
    cout << frobenius_norm(C-K) << "\n";

    int i = 38;
    auto factors = prime_factors(i);
    for (const int f : factors)
        cout << f << " ";
    cout << "\n";
    return 0;
}
