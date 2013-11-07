from math import sqrt
import numpy as np
from scipy.linalg import eig, solve, norm, inv


class StochasticLinearDiff(object):
    """
    Represents and computes various things for a model in the form
    of the canonical stochastic linear difference equation:

    .. math::

        x_{t+1} = A x_t + C w_{t+1}
    """

    def __init__(self, A, C):
        self.A = A
        self.C = C

        # Evaluate eigenvalues and vectors for use later on. Check boundedness
        evals, evecs = eig(self.A, left=False, right=True)
        self.evals, self.evecs = evals, evecs
        self.unbounded = np.abs(evals).max() > 1

    def Cx(self, j=0):
        "Covariance stationary covariance matrix"
        if not self.unbounded:
            c_x = doublej(self.A, self.C.dot(self.C.T))

            # Return if we want C_x(0)
            if j == 0:
                return c_x
            else:
                # Or evaluate C_x(abs(j))
                c_xj = np.linalg.matrix_power(self.A, abs(j)).dot(c_x)
            if j < 0:
                return c_xj.T  # transpose if j < 0
            else:
                return c_xj

        else:
            msg = 'This computation will not work because the eigenvalues'
            msg += '\nof A are not all below 1 in modulus.'
            raise ValueError(msg)

    @property
    def mu(self):
        "Covariance stationary mean"
        if self.unbounded:
            msg = 'This computation will not work because the eigenvalues {0}'
            msg += '\nof A are not all below 1 in modulus.'
            raise ValueError(msg.format(self.evals))

        # Try to get index of unit eigenvalue
        try:
            ind = np.where(self.evals == 1)[0][0]
        except IndexError:
            raise ValueError("The A matrix doesn't have any unit eigenvalues")

        # compute Stationary mean using the eigenvector for unit eigenvalue
        return self.evecs[:, ind] / self.evecs[-1, ind]


class Markov(object):
    """
    Do basic things with Markov matrices.
    """

    def __init__(self, P, verbose=False):
        self.P = P
        self.verbose = verbose

    def __repr__(self):
        msg = "Markov process with transition matrix P = \n{0}"
        return msg.format(self.P)

    def stationary_distributions(self):
        evals, l_evecs, r_evecs = eig(self.P, left=True, right=True)
        self.evals, self.l_evecs, self.r_evecs = evals, l_evecs, r_evecs
        units = np.where(evals == 1)[0]
        stationary = []
        for i, ind in enumerate(units):
            sd_name = 'sd{0}'.format(i + 1)
            sd_vec = l_evecs[:, ind]

            # Normalize to be probability vector
            sd_vec = sd_vec * (-1) if all(sd_vec <= 0) else sd_vec
            sd_vec /= sd_vec.sum()
            self.__setattr__(sd_name, sd_vec)
            stationary.append(sd_vec)
            if self.verbose:
                msg = 'Set instance variable %s for stationary distribution'
                print(msg % sd_name)
        return stationary

    def invariant_distributions(self):
        units = np.where(self.evals == 1)[0]
        invariant = []
        for i, ind in enumerate(units):
            id_name = 'id{0}'.format(i + 1)
            id_vec = self.r_evecs[:, ind]
            self.__setattr__(id_name, id_vec)
            invariant.append(id_vec)
            if self.verbose:
                msg = 'Set instance variable %s for invariant distribution'
                print(msg % id_name)
        return invariant


class SymMarkov(object):
    """
    Do basic things with Markov matrices. The matrix P that is passed
    to the constructor for this class is assumed to be a sympy matrix.
    If it isn't, then it is cast as such.
    """

    def __init__(self, P, verbose=False):
        import sympy as sym
        self.P = P if isinstance(P, sym.Matrix) else sym.Matrix(P)
        self.verbose = verbose

    def stationary_distributions(self, subs, normalize=True):
        """
        Find the stationary distributions associated with the Markov
        process, by substituting parameters into the transition matrix

        Parameters
        ==========
        subs : dist
            A dictionary of substitutions to be passed to self.P before
            doing the computation

        normalize : bool, optional(default=True)
            Whether or not the stationary distributions should be
            normalized so they sum to 1 before returning.

        Returns
        =======
        pi0s : list
            A list of stationary distributions.

        """
        # Make the substitutions
        PN = self.P.subs(subs)

        # Transpose gives left eigenvectors
        l_vecs = PN.T.eigenvects()

        # keep only unit eigenvalues around, grab the vectors
        units = filter(lambda x: x[0] == 1, l_vecs)
        pi0s = units[0][2] if len(units) != 0 else []

        # Normalize so they sum to 1
        if normalize:
            pi0s = [i / sum(i) for i in pi0s]

        return pi0s


def doublej(a1, b1, max_it=50):
    """
    Computes the infinite sum V given by

    .. math::

        V = \sum_{j=0}^{\infty} a1^j b1 a1^j'

    where a1 and b1 are each (n X n) matrices with eigenvalues whose
    moduli are bounded by unity and b1 is an (n X n) matrix.

    V is computed by using the following 'doubling algorithm'. We
    iterate to convergence on V(j) on the following recursions for
    j = 1, 2, ... starting from V(0) = b1:

    ..math::

        a1_j = a1_{j-1} a1_{j-1}
        V_j = V_{j-1} + A_{j-1} V_{j-1} a_{j-1}'

    The limiting value is returned in V
    """
    alpha0 = a1
    gamma0 = b1

    diff = 5
    n_its = 1

    while diff > 1e-15:

        alpha1 = alpha0.dot(alpha0)
        gamma1 = gamma0 + np.dot(alpha0.dot(gamma0), alpha0.T)

        diff = np.max(np.abs(gamma1 - gamma0))
        alpha0 = alpha1
        gamma0 = gamma1

        n_its += 1

        if n_its > max_it:
            raise ValueError('Exceeded maximum iterations of %i.' % (max_it) +
                             ' Check your input matrices')

    return gamma1


def doubleo(A, C, Q, R, tol=1e-15):
    """
    This function uses the "doubling algorithm" to solve the Riccati
    matrix difference equations associated with the Kalman filter.  The
    returns the gain K and the stationary covariance matrix of the
    one-step ahead errors in forecasting the state.

    The function creates the Kalman filter for the following system:

    .. math::

        x_{t+1} = A * x_t + e_{t+1}

        y_t = C * x_t + v_t

    where :math:`E e_{t+1} e_{t+1}' =  Q`, and :math:`E v_t v_t' = R`,
    and :math:`v_s' e_t = 0 \\forall s, t`.

    The function creates the observer system

    .. math::

        xx_{t+1} = A xx_t + K a_t

        y_t = C xx_t + a_t

    where K is the Kalman gain, :math:`S = E (x_t - xx_t)(x_t - xx_t)'`,
    and :math:`a_t = y_t - E[y_t| y_{t-1}, y_{t-2}, \dots ]`, and
    :math:`xx_t = E[x_t|y_{t-1},\dots]`.

    Parameters
    ----------
    A : array_like, dtype=float, shape=(n, n)
        The matrix A in the law of motion for x

    C : array_like, dtype=float, shape=(k, n)

    Q : array_like, dtype=float, shape=(n, n)

    R : array_like, dtype=float, shape=(k, k)

    tol : float, optional(default=1e-15)

    Returns
    -------
    K : array_like, dtype=float
        The Kalman gain K

    S : array_like, dtype=float
        The stationary covariance matrix of the one-step ahead errors
        in forecasting the state.

    Notes
    -----
    By using DUALITY, control problems can also be solved.
    """
    a0 = A.T
    b0 = C.T.dot(solve(R, C))
    g0 = Q
    dd = 1
    ss = max(A.shape)
    v = np.eye(ss)

    while dd > tol:
        a1 = a0.dot(solve(v + np.dot(b0, g0), a0))
        b1 = b0 + a0.dot(solve(v + np.dot(b0, g0), b0.dot(a0.T)))
        g1 = g0 + np.dot(a0.T.dot(g0), solve(v + b0.dot(g0), a0))
        k1 = np.dot(A.dot(g1), solve(np.dot(C, g1.T).dot(C.T) + R.T, C).T)
        k0 = np.dot(A.dot(g0), solve(np.dot(C, g0.T).dot(C.T) + R.T, C).T)
        a0=a1
        b0=b1
        g0=g1
        dd = np.max(k1 - k0)

    return k1, g1


def markov(T, n=100, s0=0, V=None):
    """
    Generates a simulation of the Markov chain described by a transition
    matrix.

    Parameters
    ==========
    T : array_like, dtype=float, ndim=2
        The Markov transition matrix that describes the model

    n : integer, optional(default=100)
        How many steps to simulate the chain

    s0 : int, optional(default=0)
        The initial state. Should be a value between 0 and T.shape[0]
        - 1 because it will be used as a python index.

    V : array_like, dtype=float, optional(default=range(T.shape[0]))
        The 1d array to specify numerical value associated with each
        state

    Returns
    =======
    chain : array_like, dtype=float
        The simulated state

    state : array_like, dtype=int
        The time series of state values

    """
    r, c = T.shape
    if V is None:
        V = np.arange(r)

    if r != c:
        raise ValueError('T must be a square matrix')

    _row_sums = T.sum(axis=1)
    if not all(_row_sums == 1):
        bad = np.where(_row_sums != 1)
        msg = 'All rows of T must sum to 1. Column(s) %s do not'
        raise ValueError(msg % (bad[0]))

    if V.ndim != 1:
        V = V.flatten()
        if V.size != r:
            msg = 'V must be 1-dimensional array of length %i' % (r)
            raise ValueError(msg)

    if s0 < 0 or s0 > (r - 1):
        msg = 'Value of s0 (%i) must be between 0 and T.shape[0] (%i)'
        raise ValueError(msg % (s0, r - 1))

    X = np.random.rand(n - 1)
    s = np.zeros(r)
    s[s0] = 1
    cdf = np.cumsum(T, axis=1)
    state = np.empty((r, n - 1))
    for k in range(n - 1):
        state[:, k] = s
        ppi = np.concatenate([[0.], s.dot(cdf)])
        s = (X[k] <= ppi[1:]) * (X[k] > ppi[:-1])

    chain = V.dot(state)
    return chain, state


def olrp(beta, A, B, Q, R, W=None, tol=1e-6, max_iter=1000):
    """
    Calculates F of the feedback law:

    .. math::

          U = -Fx

     that maximizes the function:

     .. math::

        \sum \{beta^t [x'Qx + u'Ru +2x'Wu] \}

     subject to

     .. math::
          x_{t+1} = A x_t + B u_t

    where x is the nx1 vector of states, u is the kx1 vector of controls

    Parameters
    ----------
    beta : float
        The discount factor from above. If there is no discounting, set
        this equal to 1.

    A : array_like, dtype=float, shape=(n, n)
        The matrix A in the law of motion for x

    B : array_like, dtype=float, shape=(n, k)
        The matrix B in the law of motion for x

    Q : array_like, dtype=float, shape=(n, n)
        The matrix Q from the objective function

    R : array_like, dtype=float, shape=(k, k)
        The matrix R from the objective function

    W : array_like, dtype=float, shape=(n, k), optional(default=0)
        The matrix W from the objective function. Represents the cross
        product terms.

    tol : float, optional(default=1e-6)
        Convergence tolerance for case when largest eigenvalue is below
        1e-5 in modulus

    max_iter : int, optional(default=1000)
        The maximum number of iterations the function will allow before
        stopping

    Returns
    -------
    F : array_like, dtype=float
        The feedback law from the equation above.

    P : array_like, dtype=float
        The steady-state solution to the associated discrete matrix
        Riccati equation

    """
    m = max(A.shape)
    rc, cb = np.atleast_2d(B).shape

    if W is None:
        W = np.zeros((m, cb))

    if np.max(np.abs(eig(R)[0])) > 1e-5:
        A = sqrt(beta) * (A - B.dot(solve(R, W.T)))
        B = sqrt(beta) * B
        Q = Q - W.dot(solve(R, W.T))

        k, s = doubleo(A.T, B.T, Q, R)

        f = k.T + solve(R, W.T)

        p = s

    else:
        p0 = -0.1 * np.eye(m)
        dd = 1
        it = 1

        for it in range(max_iter):
            f0 = solve(R + beta * B.T.dot(p0).dot(B),
                       beta * B.T.dot(p0).dot(A) + W.T)
            p1 = beta * A.T.dot(p0).dot(A) + Q - \
                (beta * A.T.dot(p0).dot(B) + W).dot(f0)
            f1 = solve(R + beta * B.T.dot(p1).dot(B),
                       beta * B.T.dot(p1).dot(A) + W.T)
            dd = np.max(f1 - f0)
            p0 = p1

            if dd > tol:
                break
        else:
            msg = 'No convergence: Iteration limit of {0} reached in OLRP'
            raise ValueError(msg.format(max_iter))

        f = f1
        p = p1

    return f, p


def ricatti(beta, A, B, R, Q, H, tol=1e-6, maxiter=1000):
    """
    Calculates F of the feedback law:

    .. math::

          U = -Fx

     that maximizes the function:

     .. math::

        \sum \{beta^t [x'Qx + u'Ru +2x'Wu] \}

     subject to

     .. math::
          x_{t+1} = A x_t + B u_t

    where x is the nx1 vector of states, u is the kx1 vector of controls

    Parameters
    ----------
    beta : float
        The discount factor from above. If there is no discounting, set
        this equal to 1.

    A : array_like, dtype=float, shape=(n, n)
        The matrix A in the law of motion for x

    B : array_like, dtype=float, shape=(n, k)
        The matrix B in the law of motion for x

    R : array_like, dtype=float, shape=(k, k)
        The matrix R from the objective function

    Q : array_like, dtype=float, shape=(n, n)
        The matrix Q from the objective function

    H : array_like, dtype=float, shape=(n, k), optional(default=0)
        The matrix W from the objective function. Represents the cross
        product terms.

    tol : float, optional(default=1e-6)
        Convergence tolerance for case when largest eigenvalue is below
        1e-5 in modulus

    max_iter : int, optional(default=1000)
        The maximum number of iterations the function will allow before
        stopping

    Returns
    -------
    F : array_like, dtype=float
        The feedback law from the equation above.

    P : array_like, dtype=float
        The steady-state solution to the associated discrete matrix
        Riccati equation

    """
    n = A.shape[0]
    k = np.ascontiguousarray(Q).shape[0]

    A, B, R, Q, H = map(np.matrix, [A, B, R, Q, H])

    A = A.reshape(n, n)
    B = B.reshape(n, k)
    Q = Q.reshape(k, k)
    R = R.reshape(n, n)
    H = H.reshape(k, n)

    # Start with an initial P matrix
    p0 = np.zeros((n, n))
    p1 = np.zeros((n, n))

    # Define some variables necessary to enter while loop
    dist = 10.
    iters = 0

    while dist > tol and iters < maxiter:
        p1 = R + beta*A.T*p0*A - ((beta*A.T*p0*B + H.T) *
                                  inv(Q + beta*B.T*p0*B) *
                                  (beta*B.T*p0*A + H))

        dist = norm(p1 - p0)
        print("Iteration is %i and norm is %.3e" % (iters, dist))
        p0 = p1

    P = p0

    F = inv((Q + beta*B.T.dot(P.dot(B)))).dot(beta*B.T.dot(P.dot(A)) + H)

    return map(np.array, [F, P])


if __name__ == '__main__':
    P = np.array([[.7, .3], [.2, .8]])
    c, s = markov(P, n=2000, V=np.array([1., 2.]))
