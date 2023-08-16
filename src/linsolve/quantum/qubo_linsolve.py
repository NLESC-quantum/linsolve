"""High-level tools for linearizing and finding chi^2 minimizing systems of equations using Quantum Routines

Solvers: QuantumLinearSolver, QuantumLogProductSolver, and QuantumLinProductSolver.

These generally follow the form:

    >>> data = {'a1*x+b1*y': np.array([5.,7]), 'a2*x+b2*y': np.array([4.,6])}
    >>> ls = LinearSolver(data, a1=1., b1=np.array([2.,3]), a2=2., b2=np.array([1.,2]))
    >>> sol = ls.solve()

where equations are passed in as a dictionary where each key is a string
describing the equation (which is parsed according to python syntax) and each
value is the corresponding "measured" value of that equation.  Variable names
in equations are checked against keyword arguments to the solver to determine
if they are provided constants or parameters to be solved for.  Parameter anmes
and solutions are return are returned as key:value pairs in ls.solve().
Parallel instances of equations can be evaluated by providing measured values
as numpy arrays.  Constants can also be arrays that comply with standard numpy
broadcasting rules.  Finally, weighting is implemented through an optional wgts
dictionary that parallels the construction of data.

LinearSolver solves linear equations of the form 'a*x + b*y + c*z'.
LogProductSolver uses logrithms to linearize equations of the form 'x*y*z'.
LinProductSolver uses symbolic Taylor expansion to linearize equations of the
form 'x*y + y*z'.

For more detail on usage, see linsolve_example.ipynb
"""

import ast
import numpy as np
from time import time 

from qiskit.quantum_info import Statevector
from ..linsolve import LinearSolver, LogProductSolver, LinProductSolver
from ..linsolve import get_name

# Monkey patch for backward compatibility:
# ast.Num deprecated in Python 3.8. Make it an alias for ast.Constant
# if it gets removed.
if not hasattr(ast, "Num"):
    ast.Num = ast.Constant


class SolverResult:

    def __init__(self):
        self.solution_vector = []
        self.true_rhs = []
        self.sol_rhs = []

    def update(self, vec, rhs, rhs_approx):
        self.solution_vector.append(vec)
        self.true_rhs.append(rhs)
        self.sol_rhs.append(rhs_approx)

    def clean_up(self):
        max_size = 0
        for cfun in self.cost_function:
            max_size = max(len(cfun), max_size)

        cleaned_cost_function = []
        for cfun in self.cost_function:
            cleaned_cost_function.append(cfun + (max_size-len(cfun)) * [0.])

        self.cost_function = cleaned_cost_function 
            
    def todict(self):
        return self.__dict__
        

class QUBOLinearSolver(LinearSolver):
    def __init__(self, solver, data, wgts={}, sparse=False, **kwargs):
        """Set up a linear system of equations of the form 1*a + 2*b + 3*c = 4.

        Parameters
        ----------
        data : dict
            maps linear equations, written as valid python-interpetable strings
            that include the variables in question, to (complex) numbers or numpy
            arrays. Variables with trailing underscores '_' are interpreted as complex
            conjugates.
        wgts : dict
            maps equation strings from data to real weights to apply to each
            equation. Weights are treated as 1/sigma^2. All equations in the data must
            have a weight if wgts is not the default, {}, which means all 1.0s.
        sparse : bool
            If True, represents A matrix sparsely (though AtA, Aty end up dense)
            May be faster for certain systems of equations.
        **kwargs: keyword arguments of constants (python variables in keys of data that
            are not to be solved for)
        """

        super().__init__(data, wgts, sparse, **kwargs)
        self.solver = solver

    @staticmethod
    def _normalization(mat, vec):
        """normalize the matrix and vector of the linear systems

        Args:
            mat (_type_): _description_
            vec (_type_): _description_
        """
        mat_norm = np.linalg.norm(mat)
        vec_norm = np.linalg.norm(vec)

        return (mat / mat_norm, vec / vec_norm), (mat_norm, vec_norm)

    @staticmethod
    def _unnormalization(sol, mat_norm, vec_norm):
        """un normalized the solution vector

        Args:
            sol (_type_): _description_
            mat_norm (_type_): _description_
            vec_norm (_type_): _description_
        """
        return sol / mat_norm * vec_norm

    def _invert_qubo(self, A, y, rcond):
        """_summary_

        Args:
            A (_type_): _description_
            y (_type_): _description_
            rcond (_type_): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        AtA, Aty = self._process_data(A, y)
        system_size, num_rhs = AtA[0].shape, y.shape[1]

        # init the outputs
        output, solver_result = [], SolverResult()
        print('QUBO Linsolve: vqls_shared %dx%d system with %d rhs' %(system_size[0], system_size[1], num_rhs))
        idx_rhs = 1

        for m, y in zip(AtA, Aty):
            tinit = time()
            if np.linalg.norm(y) == 0:
                solution_vector = np.zeros(true_size)
            else:
                (m_, y_), (m_norm, y_norm) = self._normalization(m, y)
                solution_vector = self.solver.solve(m_, y_)
                solution_vector = self._unnormalization(solution_vector, m_norm, y_norm)

                # store results
                solver_result.update( solution_vector, y, m@solution_vector)
            elapsed_time = (time() - tinit) / 60.
            
            if idx_rhs == 1:
                print(f'\t First iteration done in {elapsed_time} min.', flush=True)
                print(f'\t Estimated runtime {elapsed_time * num_rhs / 60.} hours.', flush=True)
            else:
                print(f'\t {idx_rhs} iteration done in {elapsed_time} min.', flush=False)


            # store solution vector 
            output.append(solution_vector)
            idx_rhs += 1
        return np.array(output).T, solver_result
    
    def _process_data(self, A, y):

        At = A.transpose([2, 1, 0]).conj()

        if At.shape[0] == 1:
            AtA = [np.dot(At[0], A[..., 0])] * (y.shape[-1])
            Aty = [np.dot(At[0], y[..., k]) for k in range(y.shape[-1])]
        else:
            AtA = [np.dot(At[k], A[..., k]) for k in range(y.shape[-1])]
            Aty = [np.dot(At[k], y[..., k]) for k in range(y.shape[-1])]
        
        return AtA, Aty
    
    def return_matrix(self):
        """Return the matrices
        """
        
        y = self.get_weighted_data()
        A = self.get_A()
        return self._process_data(A, y)

    def solve(self, rcond=None, mode="qubo"):
        """Compute x' = (At A)^-1 At * y, returning x' as dict of prms:values.

        Parameters
        ----------
        rcond
            cutoff ratio for singular values useed in :func:`numpy.linalg.lstsq`,
            :func:`numpy.linalg.pinv`, or (if sparse) as atol and btol in
            :func:`scipy.sparse.linalg.lsqr`
        mode : {'default', 'lsqr', 'pinv', or 'solve'},
            selects which inverter to use, unless all equations share the same A matrix,
            in which case pinv is always used:

            All of these modes are superceded if the same system of equations applies
            to all datapoints in an array.  In this case, a inverse-based method is
            used so that the inverted matrix can be re-used to solve all array indices.

        Returns
        -------
        sol
            a dictionary of solutions with variables as keys
        """
        assert mode in ["qubo"]
        if rcond is None:
            rcond = np.finfo(self.dtype).resolution
        y = self.get_weighted_data()
        if self.sparse:
            raise ValueError("Quantum solver not implemented yet for sparse matrices")
        else:
            A = self.get_A()
            assert A.ndim == 3
            x, res = self._invert_qubo(A, y, rcond)

        x.shape = x.shape[:1] + self._data_shape  # restore to shape of original data
        sol = {}
        for p in list(self.prms.values()):
            sol.update(p.get_sol(x, self.prm_order))
        return sol, res


class QUBOLogProductSolver(LogProductSolver):
    def __init__(self, solver, data, wgts={}, sparse=False, **kwargs):
        """A log-solver for systems of equations of the form a*b = 1.0.

        Parameters
        ----------
        data
            dict mapping nonlinear product equations, written as valid
            python-interpetable strings that include the variables in question, to
            (complex) numbers or numpy arrarys. Variables with trailing underscores '_'
            are interpreted as complex conjugates (e.g. x*y_ parses as x * y.conj()).
        wgts
            dict that maps equation strings from data to real weights to apply to each
            equation. Weights are treated as 1/sigma^2. All equations in the data must
            have a weight if wgts is not the default, {}, which means all 1.0s.
        sparse : bool.
            If True, represents A matrix sparsely (though  AtA, Aty end up dense).
            May be faster for certain systems of equations.
        **kwargs: keyword arguments of constants (python variables in keys of data that
            are not to be solved for)
        """
        super().__init__(data, wgts, sparse, **kwargs)

        constants = kwargs.pop("constants", kwargs)
        logamp_consts, logphs_consts = {}, {}
        for k in constants:
            c = np.log(constants[k])  # log unwraps complex circle at -pi
            logamp_consts[k], logphs_consts[k] = c.real, c.imag

        self.ls_amp = QUBOLinearSolver(
            solver,
            self.ls_amp.data,
            self.ls_amp.wgts,
            sparse=self.ls_amp.sparse,
            constants=logamp_consts,
            **kwargs,
        )

        if self.ls_phs is not None:
            self.ls_phs = QUBOLinearSolver(
                solver,
                self.ls_phs.data,
                self.ls_phs.wgts,
                sparse=self.ls_phs.sparse,
                constants=logphs_consts,
                **kwargs,
            )


# XXX make a version of linproductsolver that taylor expands in e^{a+bi} form
# see https://github.com/HERA-Team/linsolve/issues/15
class QUBOLinProductSolver(LinProductSolver):
    def __init__(self, solver, data, sol0, wgts={}, sparse=False, **kwargs):
        """Set up a nonlinear system of equations of the form a*b + c*d = 1.0.

        Linearize via Taylor expansion and solve iteratively using the Gauss-Newton
        algorithm.

        Parameters
        ----------
        data
            dict that maps nonlinear product equations, written as valid
            python-interpetable strings that include the variables in question, to
            (complex) numbers or numpy arrarys. Variables with trailing underscores '_'
            are interpreted as complex conjugates (e.g. x*y_ parses as x * y.conj()).
        sol0
            dict mapping all variables (as keyword strings) to their starting guess
            values. This is the point that is Taylor expanded around, so it must be
            relatively close to the true chi^2 minimizing solution. In the same format
            as that produced by :func:`~LogProductSolver.solve()` or
            :func:`~LinProductSolver.solve()`.
        wgts
            dict that maps equation strings from data to real weights to apply to each
            equation. Weights are treated as 1/sigma^2. All equations in the data must
            have a weight if wgts is not the default, {}, which means all 1.0s.
        sparse : bool
            If True, represents A matrix sparsely (though AtA, Aty end up dense)
            May be faster for certain systems of equations.
        **kwargs: keyword arguments of constants (python variables in keys of data that
            are not to be solved for)
        """
        # XXX make this something hard to collide with
        # see https://github.com/HERA-Team/linsolve/issues/17

        super().__init__(data, sol0, wgts, sparse, **kwargs)
        self.solver = solver

    def build_solver(self, sol0):
        """Builds a LinearSolver using the taylor expansions and all relevant constants.

        Update it with the latest solutions.
        """
        dlin, wlin = {}, {}
        for k in self.keys:
            tk = self.taylor_keys[k]
            # in theory, this will always be replaced with data - ans0 before use
            dlin[tk] = self.data[k]
            try:
                wlin[tk] = self.wgts[k]
            except KeyError:
                pass
        self.ls = QUBOLinearSolver(
            self.solver, dlin, wgts=wlin, sparse=self.sparse, constants=self.sols_kwargs
        )
        self.eq_dict = {
            eq.val: eq for eq in self.ls.eqs
        }  # maps taylor string expressions to linear equations
        # Now make sure every taylor equation has every relevant constant, even if
        # they don't appear in the derivative terms.
        for k, terms in zip(self.keys, self.all_terms):
            for term in terms:
                for t in term:
                    t_name = get_name(t)
                    if t_name in self.sols_kwargs:
                        self.eq_dict[self.taylor_keys[k]].add_const(
                            t_name, self.sols_kwargs
                        )
        self._update_solver(sol0)
