import numpy as np

def sample_points_numrange(A: np.ndarray, npoints: int):
    '''
    Sample npoints within the numerical range of matrix A.
    '''
    # A must be a square matrix.
    assert(A.shape[0] == A.shape[1])

    # Sample unit length points in C^N.
    def sample_sphere(N, npoints):
        data = np.random.randn(N, npoints, 2)
        Z = data[..., 0] + 1j * data[..., 1]
        Z /= np.linalg.norm(Z, axis=0)
        return Z

    N = A.shape[0]
    Z = sample_sphere(N, npoints)
    # ----------------------------------------------------
    # Naive approach to computing the list of z^t A z:
    # Take only the diagonal of Z^t A Z... Which is N^2
    #return np.diag(Z.conj().T @ A @ Z).tolist()
    # ----------------------------------------------------
    # Best approach is clever algebra: sum over the column of element wise product.
    return np.sum(Z.conj() * (A @ Z), axis=0).tolist()

def compute_numrange_boundary(A: np.ndarray, n_points: int):
    '''
    Find the approximate boundary of W(A) with a polyline of n_points.
    '''
    # A is a square matrix
    assert(A.shape[0] == A.shape[1])
    assert(n_points >= 0)

    if n_points == 0:
        return []

    bound_points = []
    for i in range(n_points):
        theta = 2 * i * np.pi / n_points
        A_rot = A * np.exp(theta * 1j)
        # Hermitian part of A rotated
        A_h = 0.5 * (A_rot + A_rot.T.conj())
        # Compute max eigenvalue with corresponding eigenvector
        eigenvals, eigenvecs = np.linalg.eig(A_h)
        ind = np.argmax(eigenvals)
        l_max = eigenvals[ind]
        v_max = eigenvecs[:, ind]
        v_max = v_max / (v_max.T.conj() @ v_max) # normalized

        # Add to list of boundary points
        bound_points.append(v_max.T.conj() @ A @ v_max)

    # Add first point to get a closed polyline
    bound_points.append(bound_points[0])
    return bound_points

def numrange_boundary(A: np.ndarray, n_points: int):
    '''
    Determine the boundary of the numerical range of matrix A, treating special cases.
    Falls back to `compute_numrange_boundary` for the general case.
    '''
    # A is a square matrix
    assert(A.shape[0] == A.shape[1])
    assert(n_points >= 0)

    # Pre compute conjugate transpose.
    A_t = A.conj().T

    # Check if A is Hermitian.
    if np.allclose(A, A_t):
        print("A is Hermitian")
        eig_vals = np.linalg.eigvals(A)
        # Make sure eigenvalues are all real -> imag == 0.
        assert np.allclose(eig_vals.imag, np.zeros(A.shape[0]))
        # Numerical range is the line segment connecting all eigenvalues.
        return np.linspace(eig_vals.min(), eig_vals.max(), n_points)

    # Check if A is skew Hermitian.
    if np.allclose(A, -A_t):
        print("A is skew Hermitian")
        eig_vals = np.linalg.eigvals(A)
        # Make sure eigenvalues are all imaginary -> real == 0.
        assert np.allclose(eig_vals.real, np.zeros(A.shape[0]))
        # Numerical range is the line segment connecting all eigenvalues.
        return np.linspace(eig_vals.min(), eig_vals.max(), n_points)

    # Check if A is normal.
    if np.array_equal(A @ A_t, A_t @ A):
        print("A is Normal")
        # If A is normal, its numerical range is the convex hull of its eigenvalues.
        eig_vals = np.linalg.eigvals(A)
        # eig_vals as x,y points in cartesian coordinates
        points = np.array([[x,y] for x,y in zip(eig_vals.real, eig_vals.imag)])

        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        hull_as_list = [complex(points[i,0], points[i,1]) for i in hull.vertices]
        # Add first point to get a closed polyline.
        hull_as_list.append(hull_as_list[0])
        return hull_as_list

    # If not a special case, run the general algorithm.
    return compute_numrange_boundary(A, n_points)


### Plotting utilities.

def plot_point_lists(boundary: list, interior: list, eigvalues: list):
    import matplotlib.pyplot as plt

    def imaginary_to_cartesian(points: list):
        points_array = np.array(points)
        x_data = points_array.real
        y_data = points_array.imag
        return (x_data, y_data)

    if len(boundary) > 0:
        x_bound, y_bound = imaginary_to_cartesian(boundary)
        plt.plot(x_bound, y_bound, '-', color='#0000ff', linewidth=4)

    if len(interior) > 0:
        x_interior, y_interior = imaginary_to_cartesian(interior)
        plt.plot(x_interior, y_interior, '.', color='#009900', markersize=1)

    if len(eigvalues) > 0:
        x_eig, y_eig = imaginary_to_cartesian(eigvalues)
        plt.plot(x_eig, y_eig, '*', color='#ff0000', markersize=8)

    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.axis('square')
    plt.show()

def plot_numrange(A: np.ndarray, nbound: int, nwithin: int):
    '''
    Plot numerical range of A.
    '''
    plot_point_lists(
        boundary=numrange_boundary(A, nbound),
        interior=sample_points_numrange(A, nwithin),
        eigvalues=np.linalg.eigvals(A)
    )


### Usage of the functions implemented above.
def main():
    A_ellipse = np.array([[2+1j, 0+1j], [1+0j, 1+2j]])
    plot_numrange(A_ellipse, nbound=100, nwithin=20000)

    A_circle = np.diag([np.exp(1j * 2 * np.pi * n / 10) for n in range(10)])
    plot_numrange(A_circle, nbound=100, nwithin=20000)

    def rand_complex_mat(n: int, l=1.0):
        shape = (n,n)
        return np.random.uniform(-l, l, shape) + 1.j * np.random.uniform(-l, l, shape)

    # Spectrum preserving matrix transform.
    T = rand_complex_mat(10, 1.0)
    T_inv = np.linalg.inv(T)

    A = T @ A_circle @ T_inv
    plot_numrange(A, nbound=100, nwithin=20000)

    A = A_circle
    A[0,0] += 2.0
    plot_numrange(A, nbound=100, nwithin=20000)

    A[0,8] += 2.0
    plot_numrange(A, nbound=100, nwithin=20000)

    A = rand_complex_mat(8, 1.0)
    plot_numrange(A, nbound=100, nwithin=20000)

    # Hermitian matrix example.
    A = np.array([[2, 1], [1, 2]])
    plot_numrange(A, nbound=100, nwithin=100)

    # Normal matrix example.
    A = np.array([[1,1,0], [0,1,1], [1,0,1]])
    plot_numrange(A, nbound=100, nwithin=100)

if __name__ == "__main__":
    main()
