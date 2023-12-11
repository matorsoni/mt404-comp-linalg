#include "low_rank_approx.hpp"

#include <cassert>
#include <cmath>  // std::sqrt


using namespace std;
using namespace Eigen;


tuple<MatrixXd, MatrixXd, MatrixXd> SVD_eigen(const MatrixXd& matrix)
{
    JacobiSVD<MatrixXd> svd(matrix, ComputeFullU | ComputeFullV);
    return make_tuple(svd.matrixU(), svd.singularValues().asDiagonal(), svd.matrixV());
}

tuple<MatrixXd, MatrixXd, MatrixXd> SVD(const MatrixXd& matrix)
{
    const MatrixXd M = matrix * matrix.transpose();
    const MatrixXd N = matrix.transpose() * matrix;

    EigenSolver<MatrixXd> eigen_solver(M);
    VectorXd eigenvals = eigen_solver.eigenvalues().real();
    MatrixXd U = eigen_solver.eigenvectors().real();

    EigenSolver<MatrixXd> eigen_solver2(N);
    MatrixXd V = eigen_solver2.eigenvectors().real();

    // Compute the singular values and matrix U
    MatrixXd S(matrix.rows(), matrix.cols());
    S.setZero();
    for (int i = 0; i < min(matrix.rows(), matrix.cols()); i++)
        S(i, i) = sqrt(eigenvals(i));

    return make_tuple(U, S, V);
}


vector<int> prime_factors(int n)
{
    vector<int> factors;

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

tuple<MatrixXd, MatrixXd> low_rank_approx(const MatrixXd& C, int p, int q, int r, int s)
{
    const int m = C.rows();
    const int n = C.cols();
    assert(m == p * r);
    assert(n == q * s);

    const auto [U, S, V] = SVD(C);

    const auto sqrt_s1 = sqrt(S(0,0));
    const auto u1 = sqrt_s1 * U.col(0);
    const auto v1 = sqrt_s1 * V.row(0);
    assert (u1.rows() == m);
    assert (u1.cols() == 1);
    assert (v1.rows() == 1);
    assert (v1.cols() == n);

    MatrixXd A(p,q);
    MatrixXd B(r,s);
    // Fill A
    int k = 0;
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            A(i, j) = u1(k++);
        }
    }
    // Fill B
    k = 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < s; j++) {
            B(i, j) = v1(k++);
        }
    }

    return make_tuple(A, B);
}

vector<tuple<MatrixXd, MatrixXd>> low_rank_approx_complete(const MatrixXd& matrix)
{
    vector<tuple<MatrixXd, MatrixXd>> list_of_matrices;

    const int m = matrix.rows();
    const int n = matrix.cols();

    // Letting A_pq and B_rs be the sub-matrices such that C = A x B,
    // we must compute the low rank approximation for each valid m = pr and n = qs.
    vector<tuple<int, int, int, int>> dimension_pairs;  // pq rs pairs
    const auto m_factors = prime_factors(m);
    const auto n_factors = prime_factors(n);
    int q = 1;
    for (int i : m_factors) {
        int r = 1;
        for (int j : n_factors) {
            if (q == r)
                dimension_pairs.push_back(make_tuple(m/r, q, r, n/q));
            r *= j;
        }
        q *= i;
    }

    for (const auto [P, Q, R, S] : dimension_pairs) {
        list_of_matrices.emplace_back(
            low_rank_approx(matrix, P, Q, R, S)
        );
    }

    return list_of_matrices;
}


bool matrix_are_equal(const MatrixXd& A, const MatrixXd& B, double tol) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return false;
    }

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            if (std::abs(A(i, j) - B(i, j)) > tol) {
                return false;
            }
        }
    }

    return true;
}

double spectral_norm(const MatrixXd& matrix)
{
    JacobiSVD<MatrixXd> svd(matrix, ComputeThinU | ComputeThinV);
    return svd.singularValues()(0);
}

double frobenius_norm(const MatrixXd& matrix)
{
    return matrix.norm();
}
