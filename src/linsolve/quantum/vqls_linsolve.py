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

from time import time 
import numpy as np
import numpy as np



from ..linsolve import LinearSolver, LogProductSolver, LinProductSolver
from ..linsolve import get_name

# Monkey patch for backward compatibility:
# ast.Num deprecated in Python 3.8. Make it an alias for ast.Constant
# if it gets removed.
if not hasattr(ast, "Num"):
    ast.Num = ast.Constant


class SolverResult:

    def __init__(self):
        self.cost_function = []
        self.optimal_parameters = [] 
        self.solution_vector = []
        self.true_rhs = []
        self.sol_rhs = []

    def update(self, cost, params, vec, rhs, rhs_approx):
        self.cost_function.append(cost)
        self.optimal_parameters.append(params)
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
        


class QuantumLinearSolver(LinearSolver):
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
        if solver is not None:
            if "solver_options" in kwargs:
                self.solver_options = kwargs.pop("solver_options", kwargs)
            else:
                self.solver_options = None
            self.num_qubits = solver.ansatz.num_qubits
        elif "num_qubits" in kwargs:
            self.num_qubits = kwargs["num_qubits"]
        else:
            raise ValueError('Provide ansatz or num_qubits')
        
    @staticmethod
    def post_process_vqls_solution(A, y, x):
        """Retreive the  norm and direction of the solution vector
           VQLS provides a normalized form of the solution vector
           that can also have a -1 prefactor. This routine retrieves
           the un-normalized solution vector with the correct prefactor

        Args:
            A (np.ndarray): matrix of the linear system
            y (np.ndarray): rhs of the linear system
            x (np.ndarray): proposed solution
        """

        Ax = A @ x
        normy = np.linalg.norm(y)
        normAx = np.linalg.norm(Ax)
        prefac = normy / normAx

        if np.dot(Ax * prefac, y) < 0:
            prefac *= -1
        sol = prefac * x 
        return sol

    def _reset_solver(self):
        """Reset the matrix circuits between stages
        """
        self.solver.matrix_circuits = None
        self.solver.initial_point = None

    def _reset_logger(self):
        self.solver.logger.values = []
        self.solver.logger.parameters = []

    def _check_solver_ansatz(self, size):
        """Make sure that the ansatz has the correct size and fix it if not

        Args:
            size (np.ndarray): matrix to solve
        """
        # num_qubits = self.solver.ansatz.num_qubits
        num_qubits = self.num_qubits
        min_size, max_size = 2**(num_qubits-1), 2**num_qubits
        if min_size<size<=max_size:
            pass 
            # print('keep the same ansatz with %d qubits' %num_qubits)
            # return self.solver.ansatz
        else:
            num_required_qubits = int(np.ceil(np.log2(size)))
            print('Warning: Changed ansatz size to %d qubits' %num_required_qubits)
            self.num_qubits = num_required_qubits
            self.solver.ansatz = self.solver.ansatz.__class__(num_qubits = num_required_qubits,
                                                                reps=self.solver.ansatz.reps,
                                                                entanglement=self.solver.ansatz.entanglement)
            self.solver.num_qubits = num_required_qubits

    def _process_data(self, A, y):

        At = A.transpose([2, 1, 0]).conj()

        if At.shape[0] == 1:
            AtA = [np.dot(At[0], A[..., 0])] * (y.shape[-1])
            Aty = [np.dot(At[0], y[..., k]) for k in range(y.shape[-1])]
        else:
            AtA = [np.dot(At[k], A[..., k]) for k in range(y.shape[-1])]
            Aty = [np.dot(At[k], y[..., k]) for k in range(y.shape[-1])]
        
        true_size = len(AtA[0])
        if self.solver is not None:
            self._check_solver_ansatz(true_size)
        AtA, Aty = self._pad_matrices(AtA, Aty)
                
        return AtA, Aty, true_size
    
    def _pad_matrices(self, AtA, Aty):
        """_summary_

        Args:
            AtA (_type_): _description_
            Aty (_type_): _description_
        """
        
        size_mat = len(AtA[0])
        num_mat = len(AtA)
        # full_size = 2**self.solver.ansatz.num_qubits
        full_size = 2**self.num_qubits
        if size_mat != full_size:
            
            for imat in range(num_mat):

                # pad matrix with I
                tmp = np.eye(full_size)
                tmp[:size_mat,:size_mat] = AtA[imat]
                AtA[imat] = tmp

                # pad vects with zeros
                tmp_vec = np.zeros(full_size)
                tmp_vec[:size_mat] = Aty[imat]
                Aty[imat] = tmp_vec

        return AtA, Aty

    def _invert_vqls_shared(self, A, y, rcond):
        """Use VQLS to solve the system of equation. Requires a fully constrained
        system of equation with an hermitian AtA matrix.
        rcond' is unused, but passed as an argument to match the interface of other
        _invert methods."""

        # pre process the matrix and vectors
        AtA, Aty, true_size = self._process_data(A, y)
        AtA = AtA[0]
        system_size, num_rhs = AtA.shape, y.shape[1]

        # init the outputs
        output, solver_result = [], SolverResult()
        print('VQLS Linsolve: vqls_shared %dx%d system with %d rhs' %(system_size[0], system_size[1], num_rhs))
        idx_rhs = 1

        for y in Aty:
            tinit = time()
            
            if np.linalg.norm(y) == 0:
                solution_vector = np.zeros(true_size)
            else:
                # solve the system
                sol = self.solver.solve(AtA, y)
                
                # postprocess the solution vector
                solution_vector = self.post_process_vqls_solution(AtA, y, sol.vector)

                # store results
                solver_result.update( 
                    self.solver.logger.values,
                    sol.optimal_point,
                    solution_vector,
                    y,AtA@solution_vector
                )
                # init from the last optimal point ...
                self.solver.initial_point = sol.optimal_point
            elapsed_time = (time() - tinit) / 60.
            
            if idx_rhs == 1:
                print(f'\t First iteration done in {elapsed_time} min.', flush=True)
                print(f'\t Estimated runtime {elapsed_time * num_rhs / 60.} hours.', flush=True)
            else:
                print(f'\t {idx_rhs} iteration done in {elapsed_time} min.', flush=False)


            # store solution vector 
            output.append(solution_vector[:true_size])

            # reset the logger
            self._reset_logger()
            idx_rhs += 1

        output = np.array(output).T
        self._reset_solver()
        solver_result.clean_up()
        return output, solver_result

    def _invert_vqls(self, A, y, rcond):
        """Use VQLS to solve the system of equation. Requires a fully constrained
        system of equation with an hermitian AtA matrix.
        rcond' is unused, but passed as an argument to match the interface of other
        _invert methods."""

        AtA, Aty, true_size = self._process_data(A, y)

        output = []

        for m, y in zip(AtA, Aty):
            sol = self.solver.solve(m, y)
            solution_vector = self.post_process_vqls_solution(m, y, sol.vector)
            output.append(solution_vector[:true_size])

        self._reset_solver()
        return np.array(output).T

    def return_matrix(self):
        """Return the matrices
        """
        
        y = self.get_weighted_data()
        A = self.get_A()
        return self._process_data(A, y)


    def solve(self, rcond=None, mode="vqls"):
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

            * 'default': alias for 'pinv'.
            * 'lsqr': uses numpy.linalg.lstsq to do an inversion-less solve.  Usually
              the fastest solver.
            * 'solve': uses numpy.linalg.solve to do an inversion-less solve.  Fastest,
              but only works for fully constrained systems of equations.
            * 'pinv': uses numpy.linalg.pinv to perform a pseudo-inverse and then
              solves. Can sometimes be more numerically stable (but slower) than 'lsqr'.

            All of these modes are superceded if the same system of equations applies
            to all datapoints in an array.  In this case, a inverse-based method is
            used so that the inverted matrix can be re-used to solve all array indices.

        Returns
        -------
        sol
            a dictionary of solutions with variables as keys
        """
        assert mode in ["vqls"]
        if rcond is None:
            rcond = np.finfo(self.dtype).resolution
        y = self.get_weighted_data()
        if self.sparse:
            raise ValueError("Quantum solver not implemented yet for sparse matrices")
        else:
            A = self.get_A()
            Ashape = self._A_shape()
            assert A.ndim == 3
            if Ashape[-1] == 1 and y.shape[-1] > 1:  # can reuse inverse
                x, res = self._invert_vqls_shared(A, y, rcond)
            else:  # we can't reuse inverses
                x, res = self._invert_vqls(A, y, rcond)

        x.shape = x.shape[:1] + self._data_shape  # restore to shape of original data
        sol = {}
        for p in list(self.prms.values()):
            sol.update(p.get_sol(x, self.prm_order))
        return sol, res


class QuantumLogProductSolver(LogProductSolver):
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

        self.solver = solver
        if solver is not None:
            self.num_qubits = solver.ansatz.num_qubits
        elif "num_qubits" in kwargs:
            self.num_qubits = kwargs["num_qubits"]
        else:
            raise ValueError('Provide ansatz or num_qubits')


        self.ls_amp = QuantumLinearSolver(
            solver,
            self.ls_amp.data,
            self.ls_amp.wgts,
            sparse=self.ls_amp.sparse,
            constants=logamp_consts,
            **kwargs,
        )

        if self.ls_phs is not None:
            self.ls_phs = QuantumLinearSolver(
                solver,
                self.ls_phs.data,
                self.ls_phs.wgts,
                sparse=self.ls_phs.sparse,
                constants=logphs_consts,
                **kwargs,
            )

    def return_matrix(self):
        """return the data of the linear systems
        """
        if self.ls_phs is not None:
            return self.ls_amp.return_matrix(), self.ls_phs.return_matrix() 
        
        return self.ls_amp.return_matrix()

# XXX make a version of linproductsolver that taylor expands in e^{a+bi} form
# see https://github.com/HERA-Team/linsolve/issues/15
class QuantumLinProductSolver(LinProductSolver):
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
        self.ls = QuantumLinearSolver(
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
