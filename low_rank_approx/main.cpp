#include <iostream>
#include <vector>

#include "low_rank_approx.hpp"

using namespace std;
using namespace Eigen;

int main() {
    vector<MatrixXd> test_matrices;
    test_matrices.emplace_back(MatrixXd::Random(2, 2));
    test_matrices.emplace_back(MatrixXd::Random(6, 4));
    test_matrices.emplace_back(MatrixXd::Random(8, 9));
    test_matrices.emplace_back(MatrixXd::Random(14, 24));

    for (const auto& matrix : test_matrices) {
        const auto approximations = low_rank_approx_complete(matrix);
        for (const auto [A, B] : approximations) {
            cout << "======================================\n";
            cout << "P,Q == " << A.rows() << "," << A.cols() << "\n";
            cout << "R,S == " << B.rows() << "," << B.cols() << "\n";

            const auto K = kroneckerProduct(A, B);
            cout << "AxB:\n" << K << "\n\n";

            cout << "Spectral  ||C - AxB|| : " << spectral_norm(matrix-K) << "\n";
            cout << "Frobenius ||C - AxB|| : "<< frobenius_norm(matrix-K) << "\n\n";
        }
    }


    return 0;
}
