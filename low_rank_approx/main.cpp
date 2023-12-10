#include <iostream>
#include <tuple>
#include <vector>

#include "cassert"

#include "Eigen/Dense"
#include "Eigen/KroneckerProduct"
using Eigen::MatrixXd;

// SVD implemented by Eigen
std::tuple<MatrixXd, MatrixXd, MatrixXd> SVD_eigen(const MatrixXd& matrix)
{
    Eigen::JacobiSVD<MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return std::make_tuple(svd.matrixU(), svd.singularValues().asDiagonal(), svd.matrixV());
}

// TODO: My SVD
std::tuple<MatrixXd, MatrixXd, MatrixXd> SVD(const MatrixXd& matrix)
{
    const int rows = matrix.rows();
    const int cols = matrix.cols();
    return std::make_tuple(MatrixXd(rows, rows), MatrixXd(rows, cols), MatrixXd(cols, cols));
}

// TODO: eigenvalues e vectors of simetric matrices???


// Compute the factorization of integer n, not including 1.
std::vector<int> prime_factors(int n)
{
    std::vector<int> factors;

    // Factor out 2.
    while (n % 2 == 0) {
        factors.push_back(2);
        n /= 2;
    }

    // Factor out odd numbers.
    for (int i = 3; i * i <= n; i += 2) {
        while (n % i == 0) {
            factors.push_back(i);
            n /= i;
        }
    }

    // Add the remaining n if it is a prime greater than 2.
    if (n > 2) {
        factors.push_back(n);
    }

    return factors;
}

// TODO: compute low rank approx
// Approximate C by kronecker product of matrices A_pq and B_rs
std::tuple<MatrixXd, MatrixXd> low_rank_approx(const MatrixXd& C, int p, int q, int r, int s)
{
    const int m = C.rows();
    const int n = C.cols();
    assert(m == p * r);
    assert(n == q * s);

    MatrixXd A(p,q);
    MatrixXd B(r,s);
    auto C_ = Eigen::kroneckerProduct(A,B);
    std::cout << C_ << "\n";
    std::cout << C_.rows() << "\n";

    return std::make_tuple(A, B);
}

std::vector<std::tuple<MatrixXd, MatrixXd>> low_rank_approx_complete(const MatrixXd& matrix)
{
    std::vector<std::tuple<MatrixXd, MatrixXd>> list_of_matrices;

    const int m = matrix.rows();
    const int n = matrix.cols();

    // Letting A_pq and B_rs be the sub-matrices such that C = A x B,
    // we must check if m = pr and n = qs admits non-trivial solutions (p=q=1 or r=s=1).
    const auto m_factors = prime_factors(m);
    const auto n_factors = prime_factors(n);
    if (m_factors.size() == 1 || n_factors.size() == 1) {
        // If only prime factor is the umber itself, one of p,q,r,s must be 1 -> return empty list
        return list_of_matrices;
    }

    // for each pair ....
    // lowrankApprrox (); list_of_matrices.append()

    return list_of_matrices;
}


int main() {
    // Display the original matrix, U, Sigma, and V
    auto [U, sigma, V] = SVD_eigen(matrix);
    std::cout << "Original Matrix:\n" << matrix << "\n\n";
    std::cout << "U Matrix:\n" << U << "\n\n";
    std::cout << "Singular Values (Sigma):\n" << sigma << "\n\n";
    std::cout << "V Matrix:\n" << V << "\n";

    std::cout << U.col(0).norm() << "\n";

    MatrixXd C = MatrixXd::Random(6, 4);
    low_rank_approx(matrix, 3, 2, 2, 2);

    int i = 38;
    auto factors = prime_factors(i);
    for (const int f : factors)
        std::cout << f << " ";
    std::cout << "\n";
    return 0;
}
