#include <tuple>
#include <vector>

#include "Eigen/Dense"

#include "Eigen/KroneckerProduct"

// SVD implemented by Eigen
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> SVD_eigen(const Eigen::MatrixXd& matrix);

// TODO: My SVD
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> SVD(const Eigen::MatrixXd& matrix);

// Compute the factorization of integer n, not including 1.
std::vector<int> prime_factors(int n);

// Approximate C by kronecker product of matrices A_pq and B_rs
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> low_rank_approx(const Eigen::MatrixXd& C, int p, int q, int r, int s);

std::vector<std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>> low_rank_approx_complete(const Eigen::MatrixXd& matrix);

// Determine if two matrices are almost equal element-wise.
bool matrix_are_equal(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, double tol = 1e-6);

double spectral_norm(const Eigen::MatrixXd& matrix);
double frobenius_norm(const Eigen::MatrixXd& matrix);
