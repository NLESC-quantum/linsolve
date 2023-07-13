import pytest
import linsolve
import numpy as np
import ast, io, sys

class TestLinSolve():

    def test_ast_getterms(self):
        n = ast.parse('x+y',mode='eval')
        terms = linsolve.ast_getterms(n)
        assert terms == [['x'],['y']]
        n = ast.parse('x-y',mode='eval')
        terms = linsolve.ast_getterms(n)
        assert terms == [['x'],[-1,'y']]
        n = ast.parse('3*x-y',mode='eval')
        terms = linsolve.ast_getterms(n)
        assert terms, [[3,'x'],[-1,'y']]

    def test_unary(self):
        n = ast.parse('-x+y',mode='eval')
        terms = linsolve.ast_getterms(n)
        assert terms == [[-1,'x'],['y']]

    def test_multiproducts(self):
        n = ast.parse('a*x+a*b*c*y',mode='eval')
        terms = linsolve.ast_getterms(n)
        assert terms == [['a','x'],['a','b','c','y']]
        n = ast.parse('-a*x+a*b*c*y',mode='eval')
        terms = linsolve.ast_getterms(n)
        assert terms == [[-1,'a','x'],['a','b','c','y']]
        n = ast.parse('a*x-a*b*c*y',mode='eval')
        terms = linsolve.ast_getterms(n)
        assert terms == [['a','x'],[-1,'a','b','c','y']]

    def test_taylorexpand(self):
        terms = linsolve.taylor_expand([['x','y','z']],prepend='d')
        assert terms == [['x','y','z'],['dx','y','z'],['x','dy','z'],['x','y','dz']]
        terms = linsolve.taylor_expand([[1,'y','z']],prepend='d')
        assert terms == [[1,'y','z'],[1,'dy','z'],[1,'y','dz']]
        terms = linsolve.taylor_expand([[1,'y','z']],consts={'y':3}, prepend='d')
        assert terms == [[1,'y','z'],[1,'y','dz']]

    def test_verify_weights(self):
        assert linsolve.verify_weights({},['a']) == {'a':1}
        assert linsolve.verify_weights(None,['a']) == {'a':1}
        assert linsolve.verify_weights({'a':10.0},['a']) == {'a': 10.0}
        with pytest.raises(AssertionError):
            linsolve.verify_weights({'a':1.0+1.0j}, ['a'])
        with pytest.raises(AssertionError):
            linsolve.verify_weights({'a':1.0}, ['a', 'b'])

    def test_infer_dtype(self):
        assert linsolve.infer_dtype([1.,2.]) == np.float32
        assert linsolve.infer_dtype([3,4]) == np.float32
        assert linsolve.infer_dtype([np.float32(1),4]) == np.float32
        assert linsolve.infer_dtype([np.float64(1),4]) == np.float64
        assert linsolve.infer_dtype([np.float32(1),4j]) == np.complex64
        assert linsolve.infer_dtype([np.float64(1),4j]) == np.complex128
        assert linsolve.infer_dtype([np.complex64(1),4j]) == np.complex64
        assert linsolve.infer_dtype([np.complex64(1),4.]) == np.complex64
        assert linsolve.infer_dtype([np.complex128(1),np.float64(4.)]) == np.complex128
        assert linsolve.infer_dtype([np.complex64(1),np.float64(4.)]) == np.complex128
        assert linsolve.infer_dtype([np.complex64(1),np.int32(4.)]) == np.complex64
        assert linsolve.infer_dtype([np.complex64(1),np.int64(4.)]) == np.complex64

class TestLinearEquation():

    def test_basics(self):
        le = linsolve.LinearEquation('x+y')
        assert le.terms == [['x'],['y']]
        assert le.consts == {}
        assert len(le.prms) == 2
        le = linsolve.LinearEquation('x-y')
        assert le.terms == [['x'],[-1,'y']]
        le = linsolve.LinearEquation('a*x+b*y',a=1,b=2)
        assert le.terms == [['a','x'],['b','y']]
        assert 'a' in le.consts
        assert 'b' in le.consts
        assert len(le.prms) == 2
        le = linsolve.LinearEquation('a*x-b*y',a=1,b=2)
        assert le.terms == [['a','x'],[-1,'b','y']]

    def test_more(self):
        consts = {'g5':1,'g1':1}
        for k in ['g5*bl95', 'g1*bl111', 'g1*bl103']:
            le = linsolve.LinearEquation(k,**consts)
        le.terms[0][0][0] == 'g'

    def test_unary(self):
        le = linsolve.LinearEquation('-a*x-b*y',a=1,b=2)
        assert le.terms, [[-1,'a','x'],[-1,'b','y']]

    def test_order_terms(self):
        le = linsolve.LinearEquation('x+y')
        terms = [[1,1,'x'],[1,1,'y']]
        assert terms == le.order_terms([[1,1,'x'],[1,1,'y']])
        terms2 = [[1,1,'x'],[1,'y',1]]
        assert terms == le.order_terms([[1,1,'x'],[1,'y',1]])
        le = linsolve.LinearEquation('a*x-b*y',a=2,b=4)
        terms = [[1,'a','x'],[1,'b','y']]
        assert terms == le.order_terms([[1,'a','x'],[1,'b','y']])
        terms2 = [[1,'x','a'],[1,'b','y']]
        assert terms == le.order_terms([[1,'x','a'],[1,'b','y']])
        le = linsolve.LinearEquation('g5*bl95+g1*bl111',g5=1,g1=1)
        terms = [['g5','bl95'],['g1','bl111']]
        assert terms == le.order_terms([['g5','bl95'],['g1','bl111']])

    def test_term_check(self):
        le = linsolve.LinearEquation('a*x-b*y',a=2,b=4)
        terms = [[1,'a','x'],[1,'b','y']]
        assert terms == le.order_terms([[1,'a','x'],[1,'b','y']])
        terms4 = [['c','x','a'],[1,'b','y']]
        with pytest.raises(AssertionError):
            le.order_terms(terms4)
        terms5 = [[1,'a','b'],[1,'b','y']]
        with pytest.raises(AssertionError):
            le.order_terms(terms5)

    def test_eval(self):
        le = linsolve.LinearEquation('a*x-b*y',a=2,b=4)
        sol = {'x':3, 'y':7}
        assert 2*3-4*7 == le.eval(sol)
        sol = {'x':3*np.ones(4), 'y':7*np.ones(4)}
        np.testing.assert_equal(2*3-4*7, le.eval(sol))
        le = linsolve.LinearEquation('x_-y')
        sol = {'x':3+3j*np.ones(10), 'y':7+2j*np.ones(10)}
        ans = np.conj(sol['x']) - sol['y']
        np.testing.assert_equal(ans, le.eval(sol))
        

class TestLinearSolver():

    def setup(self):
        self.sparse = False
        eqs = ['x+y','x-y']
        x,y = 1,2
        d,w = {}, {}
        for eq in eqs:
            d[eq],w[eq] = eval(eq), 1.
        self.ls = linsolve.LinearSolver(d,w,sparse=self.sparse)

    def test_basics(self):
        assert len(self.ls.prms) == 2
        assert len(self.ls.eqs) == 2
        assert self.ls.eqs[0].terms == [['x'],['y']]
        assert self.ls.eqs[1].terms == [['x'],[-1,'y']]

    def test_get_A(self):
        self.ls.prm_order = {'x':0,'y':1} # override random default ordering
        A = self.ls.get_A()
        assert A.shape == (2,2,1)
        #np.testing.assert_equal(A.todense(), np.array([[1.,1],[1.,-1]]))
        np.testing.assert_equal(A, np.array([[[1.], [1]],[[1.],[-1]]]))

    #def test_get_AtAiAt(self):
    #    self.ls.prm_order = {'x':0,'y':1} # override random default ordering
    #    AtAiAt = self.ls.get_AtAiAt().squeeze()
    #    #np.testing.assert_equal(AtAiAt.todense(), np.array([[.5,.5],[.5,-.5]]))
    #    #np.testing.assert_equal(AtAiAt, np.array([[.5,.5],[.5,-.5]]))
    #    measured = np.array([[3.],[-1]])
    #    x,y = AtAiAt.dot(measured).flatten()
    #    self.assertAlmostEqual(x, 1.)
    #    self.assertAlmostEqual(y, 2.)

    def test_solve(self):
        sol = self.ls.solve()
        np.testing.assert_almost_equal(sol['x'], 1.)
        np.testing.assert_almost_equal(sol['y'], 2.)

    def test_solve_modes(self):
        for mode in ['default','lsqr','pinv','solve']:
            sol = self.ls.solve(mode=mode)
            np.testing.assert_almost_equal(sol['x'], 1.)
            np.testing.assert_almost_equal(sol['y'], 2.)

    def test_solve_arrays(self):
        # range of 1 to 101 prevents "The exact solution is  x = 0" printouts
        x = np.arange(1,101,dtype=np.float64); x.shape = (10,10)
        y = np.arange(1,101,dtype=np.float64); y.shape = (10,10)
        eqs = ['2*x+y','-x+3*y']
        d,w = {}, {}
        for eq in eqs:
            d[eq],w[eq] = eval(eq), 1.
        ls = linsolve.LinearSolver(d,w, sparse=self.sparse)
        sol = ls.solve()
        np.testing.assert_almost_equal(sol['x'], x)
        np.testing.assert_almost_equal(sol['y'], y)

    def test_solve_arrays_modes(self):
        # range of 1 to 101 prevents "The exact solution is  x = 0" printouts
        x = np.arange(1,101,dtype=np.float64); x.shape = (10,10)
        y = np.arange(1,101,dtype=np.float64); y.shape = (10,10)
        eqs = ['2*x+y','-x+3*y']
        d,w = {}, {}
        for eq in eqs:
            d[eq],w[eq] = eval(eq), 1.
        ls = linsolve.LinearSolver(d,w, sparse=self.sparse)
        for mode in ['default','lsqr','pinv','solve']:
            sol = ls.solve(mode=mode)
            np.testing.assert_almost_equal(sol['x'], x)
            np.testing.assert_almost_equal(sol['y'], y)

    def test_A_shape(self):
        # range of 1 to 11 prevents "The exact solution is  x = 0" printouts
        consts = {'a':np.arange(1,11), 'b':np.zeros((1,10))}
        ls = linsolve.LinearSolver({'a*x+b*y':0.},{'a*x+b*y':1},**consts)
        assert ls._A_shape() == (1,2,10*10)

    def test_const_arrays(self):
        x,y = 1.,2.
        a = np.array([3.,4,5])
        b = np.array([1.,2,3])
        eqs = ['a*x+y','x+b*y']
        d,w = {}, {}
        for eq in eqs: d[eq],w[eq] = eval(eq), 1.
        ls = linsolve.LinearSolver(d,w,a=a,b=b, sparse=self.sparse)
        sol = ls.solve()
        np.testing.assert_almost_equal(sol['x'], x*np.ones(3,dtype=np.float64))
        np.testing.assert_almost_equal(sol['y'], y*np.ones(3,dtype=np.float64))

    def test_wgt_arrays(self):
        x,y = 1.,2.
        a,b = 3.,1.
        eqs = ['a*x+y','x+b*y']
        d,w = {}, {}
        for eq in eqs:
            d[eq],w[eq] = eval(eq), np.ones(4)
        ls = linsolve.LinearSolver(d,w,a=a,b=b, sparse=self.sparse)
        sol = ls.solve()
        np.testing.assert_almost_equal(sol['x'], x*np.ones(4,dtype=np.float64))
        np.testing.assert_almost_equal(sol['y'], y*np.ones(4,dtype=np.float64))

    def test_wgt_const_arrays(self):
        x,y = 1.,2.
        a,b = 3.*np.ones(4),1.
        eqs = ['a*x+y','x+b*y']
        d,w = {}, {}
        for eq in eqs:
            d[eq],w[eq] = eval(eq)*np.ones(4), np.ones(4)
        ls = linsolve.LinearSolver(d,w,a=a,b=b, sparse=self.sparse)
        sol = ls.solve()
        np.testing.assert_almost_equal(sol['x'], x*np.ones(4,dtype=np.float64))
        np.testing.assert_almost_equal(sol['y'], y*np.ones(4,dtype=np.float64))

    def test_nonunity_wgts(self):
        x,y = 1.,2.
        a,b = 3.*np.ones(4),1.
        eqs = ['a*x+y','x+b*y']
        d,w = {}, {}
        for eq in eqs: d[eq],w[eq] = eval(eq)*np.ones(4), 2*np.ones(4)
        ls = linsolve.LinearSolver(d,w,a=a,b=b, sparse=self.sparse)
        sol = ls.solve()
        np.testing.assert_almost_equal(sol['x'], x*np.ones(4,dtype=np.float64))
        np.testing.assert_almost_equal(sol['y'], y*np.ones(4,dtype=np.float64))

    def test_eval(self):
        x,y = 1.,2.
        a,b = 3.*np.ones(4),1.
        eqs = ['a*x+y','x+b*y']
        d,w = {}, {}
        for eq in eqs:
            d[eq],w[eq] = eval(eq)*np.ones(4), np.ones(4)
        ls = linsolve.LinearSolver(d,w,a=a,b=b, sparse=self.sparse)
        sol = ls.solve()
        np.testing.assert_almost_equal(sol['x'], x*np.ones(4,dtype=np.float64))
        np.testing.assert_almost_equal(sol['y'], y*np.ones(4,dtype=np.float64))
        result = ls.eval(sol)
        for eq in d:
            np.testing.assert_almost_equal(d[eq], result[eq])
        result = ls.eval(sol, 'a*x+b*y')
        np.testing.assert_almost_equal(3*1+1*2, list(result.values())[0])

    def test_chisq(self):
        x = 1.
        d = {'x':1, 'a*x':2}
        ls = linsolve.LinearSolver(d,a=1.0, sparse=self.sparse)
        sol = ls.solve()
        chisq = ls.chisq(sol)
        np.testing.assert_almost_equal(chisq, .5)
        x = 1.
        d = {'x':1, '1.0*x':2}
        ls = linsolve.LinearSolver(d, sparse=self.sparse)
        sol = ls.solve()
        chisq = ls.chisq(sol)
        np.testing.assert_almost_equal(chisq, .5)
        x = 1.
        d = {'1*x': 2.0, 'x': 1.0}
        w = {'1*x': 1.0, 'x': .5}
        ls = linsolve.LinearSolver(d, wgts=w, sparse=self.sparse)
        sol = ls.solve()
        chisq = ls.chisq(sol)
        np.testing.assert_almost_equal(sol['x'], 5.0/3.0, 6)
        np.testing.assert_almost_equal(ls.chisq(sol), 1.0/3.0)

    def test_dtypes(self):
        ls = linsolve.LinearSolver({'x_': 1.0+1.0j}, sparse=self.sparse)
        # conjugation should trigger re_im_split, splitting the
        # complex64 type into two float32 types
        assert ls.dtype == np.float32
        assert type(ls.solve()['x']) == np.complex64

        ls = linsolve.LinearSolver({'x': 1.0+1.0j}, sparse=self.sparse)
        assert ls.dtype == np.complex64
        assert type(ls.solve()['x']) == np.complex64

        ls = linsolve.LinearSolver({'x_': np.ones(1,dtype=np.complex64)[0]}, sparse=self.sparse)
        # conjugation should trigger re_im_split, splitting the
        # complex64 type into two float32 types
        assert ls.dtype == np.float32
        assert type(ls.solve()['x']) == np.complex64

        ls = linsolve.LinearSolver({'x': np.ones(1,dtype=np.complex64)[0]}, sparse=self.sparse)
        assert ls.dtype,np.complex64
        assert type(ls.solve()['x']) == np.complex64

        ls = linsolve.LinearSolver({'c*x': np.array(1.0, dtype=np.float32)}, c=1.0+1.0j, sparse=self.sparse)
        assert ls.dtype == np.complex64
        assert type(ls.solve()['x']) == np.complex64

        d = {'c*x': np.ones(1,dtype=np.float32)[0]}
        wgts = {'c*x': np.ones(1,dtype=np.float64)[0]}
        c = np.ones(1,dtype=np.float32)[0]
        ls = linsolve.LinearSolver(d, wgts=wgts, c=c, sparse=self.sparse)
        assert ls.dtype == np.float64
        assert type(ls.solve()['x']) == np.float64

        d = {'c*x': np.ones(1,dtype=np.float32)[0]}
        wgts = {'c*x': np.ones(1,dtype=np.float32)[0]}
        c = np.ones(1,dtype=np.float32)[0]
        ls = linsolve.LinearSolver(d, wgts=wgts, c=c, sparse=self.sparse)
        assert ls.dtype == np.float32
        assert type(ls.solve()['x']) == np.float32

    def test_degen_sol(self):
        # test how various solvers deal with degenerate solutions
        d = {'x+y': 1., '2*x+2*y': 2.}
        ls = linsolve.LinearSolver(d, sparse=self.sparse)
        for mode in ('pinv', 'lsqr'):
            sol = ls.solve(mode=mode)
            np.testing.assert_almost_equal(sol['x'] + sol['y'], 1., 6)
        with pytest.raises(np.linalg.LinAlgError):
            ls.solve(mode='solve')

class TestLinearSolverSparse(TestLinearSolver):

    def setup(self):
        self.sparse = True
        eqs = ['x+y','x-y']
        x,y = 1,2
        d,w = {}, {}
        for eq in eqs: d[eq],w[eq] = eval(eq), 1.
        self.ls = linsolve.LinearSolver(d,w,sparse=self.sparse)


class TestLogProductSolver():

    def setup(self):
        self.sparse=False

    def test_init(self):
        x,y,z = np.exp(1.+0j), np.exp(2.), np.exp(3.)
        keys = ['x*y*z', 'x*y', 'y*z']
        d,w = {}, {}
        for k in keys: d[k],w[k] = eval(k), 1.
        ls = linsolve.LogProductSolver(d,w,sparse=self.sparse)
        for k in ls.ls_phs.data:
            np.testing.assert_equal(ls.ls_phs.data[k], 0)
        x,y,z = 1.,2.,3.
        for k in ls.ls_amp.data:
            np.testing.assert_equal(eval(k), ls.ls_amp.data[k])

    def test_conj(self):
        x,y = 1+1j, 2+2j
        d,w = {}, {}
        d['x*y_'] = x * y.conjugate()
        d['x_*y'] = x.conjugate() * y
        d['x*y'] = x * y
        d['x_*y_'] = x.conjugate() * y.conjugate()
        for k in d: w[k] = 1.
        ls = linsolve.LogProductSolver(d,w,sparse=self.sparse)
        assert len(ls.ls_amp.data) == 4
        for k in ls.ls_amp.data:
            assert eval(k) == 3+3j # make sure they are all x+y
            assert k.replace('1','-1') in ls.ls_phs.data

    def test_solve(self):
        x,y,z = np.exp(1.+1j), np.exp(2.+2j), np.exp(3.+3j)
        keys = ['x*y*z', 'x*y', 'y*z']
        d,w = {}, {}
        for k in keys: d[k],w[k] = eval(k), 1.
        ls = linsolve.LogProductSolver(d,w,sparse=self.sparse)
        sol = ls.solve()
        for k in sol:
            np.testing.assert_almost_equal(sol[k], eval(k))
    def test_conj_solve(self):
        x,y = np.exp(1.+2j), np.exp(2.+1j)
        d,w = {'x*y_':x*y.conjugate(), 'x':x}, {}
        for k in d: w[k] = 1.
        ls = linsolve.LogProductSolver(d,w,sparse=self.sparse)
        sol = ls.solve()
        for k in sol:
            np.testing.assert_almost_equal(sol[k], eval(k))
    def test_no_abs_phs_solve(self):
        x,y,z = 1.+1j, 2.+2j, 3.+3j
        d,w = {'x*y_':x*y.conjugate(), 'x*z_':x*z.conjugate(), 'y*z_':y*z.conjugate()}, {}
        for k in list(d.keys()): w[k] = 1.
        ls = linsolve.LogProductSolver(d,w,sparse=self.sparse)
        # some ridiculousness to avoid "The exact solution is  x = 0" prints
        save_stdout = sys.stdout
        sys.stdout = io.StringIO()
        sol = ls.solve()
        sys.stdout = save_stdout
        x,y,z = sol['x'], sol['y'], sol['z']
        np.testing.assert_almost_equal(np.angle(x*y.conjugate()), 0.)
        np.testing.assert_almost_equal(np.angle(x*z.conjugate()), 0.)
        np.testing.assert_almost_equal(np.angle(y*z.conjugate()), 0.)
        # check projection of degenerate mode
        np.testing.assert_almost_equal(np.angle(x), 0.)
        np.testing.assert_almost_equal(np.angle(y), 0.)
        np.testing.assert_almost_equal(np.angle(z), 0.)
    def test_dtype(self):
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            x,y,z = np.exp(1.), np.exp(2.), np.exp(3.)
            keys = ['x*y*z', 'x*y', 'y*z']
            d,w = {}, {}
            for k in keys:
                d[k] = eval(k).astype(dtype)
                w[k] = np.float32(1.)
            ls = linsolve.LogProductSolver(d,w,sparse=self.sparse)
            # some ridiculousness to avoid "The exact solution is  x = 0" prints
            save_stdout = sys.stdout
            sys.stdout = io.StringIO()
            sol = ls.solve()
            sys.stdout = save_stdout
            for k in sol:
                assert sol[k].dtype == dtype

class TestLogProductSolverSparse(TestLogProductSolver):
    
    def setup(self):
        self.sparse=True


class TestLinProductSolver():
    def setup(self):
        self.sparse=False
    def test_init(self):
        x,y,z = 1.+1j, 2.+2j, 3.+3j
        d,w = {'x*y_':x*y.conjugate(), 'x*z_':x*z.conjugate(), 'y*z_':y*z.conjugate()}, {}
        for k in list(d.keys()): w[k] = 1.
        sol0 = {}
        for k in 'xyz': sol0[k] = eval(k)+.01
        ls = linsolve.LinProductSolver(d,sol0,w,sparse=self.sparse)
        x,y,z = 1.,1.,1.
        x_,y_,z_ = 1.,1.,1.
        dx = dy = dz = .001
        dx_ = dy_ = dz_ = .001
        for k in ls.ls.keys:
            np.testing.assert_almost_equal(eval(k), 0.002)
        assert len(ls.ls.prms) == 3

    def test_real_solve(self):
        x,y,z = 1., 2., 3.
        keys = ['x*y', 'x*z', 'y*z']
        d,w = {}, {}
        for k in keys: d[k],w[k] = eval(k), 1.
        sol0 = {}
        for k in 'xyz': sol0[k] = eval(k)+.01
        ls = linsolve.LinProductSolver(d,sol0,w,sparse=self.sparse)
        sol = ls.solve()
        for k in sol:
            np.testing.assert_almost_equal(sol[k], eval(k), 4)

    def test_single_term(self):
        x,y,z = 1., 2., 3.
        keys = ['x*y', 'x*z', '2*z']
        d,w = {}, {}
        for k in keys: d[k],w[k] = eval(k), 1.
        sol0 = {}
        for k in 'xyz': sol0[k] = eval(k)+.01
        ls = linsolve.LinProductSolver(d,sol0,w,sparse=self.sparse)
        sol = ls.solve()
        for k in sol:
            np.testing.assert_almost_equal(sol[k], eval(k), 4)

    def test_complex_solve(self):
        x,y,z = 1+1j, 2+2j, 3+2j
        keys = ['x*y', 'x*z', 'y*z']
        d,w = {}, {}
        for k in keys: d[k],w[k] = eval(k), 1.
        sol0 = {}
        for k in 'xyz': sol0[k] = eval(k)+.01
        ls = linsolve.LinProductSolver(d,sol0,w,sparse=self.sparse)
        sol = ls.solve()
        for k in sol:
            np.testing.assert_almost_equal(sol[k], eval(k), 4)

    def test_complex_conj_solve(self):
        x,y,z = 1.+1j, 2.+2j, 3.+3j
        d,w = {'x*y_':x*y.conjugate(), 'x*z_':x*z.conjugate(), 'y*z_':y*z.conjugate()}, {}
        for k in list(d.keys()): w[k] = 1.
        sol0 = {}
        for k in 'xyz': sol0[k] = eval(k) + .01
        ls = linsolve.LinProductSolver(d,sol0,w,sparse=self.sparse)
        ls.prm_order = {'x':0,'y':1,'z':2}
        _, sol = ls.solve_iteratively(mode='lsqr') # XXX fails for pinv
        x,y,z = sol['x'], sol['y'], sol['z']
        np.testing.assert_almost_equal(x*y.conjugate(), d['x*y_'], 3)
        np.testing.assert_almost_equal(x*z.conjugate(), d['x*z_'], 3)
        np.testing.assert_almost_equal(y*z.conjugate(), d['y*z_'], 3)

    def test_complex_array_solve(self):
        x = np.arange(30, dtype=np.complex128); x.shape = (3,10)
        y = np.arange(30, dtype=np.complex128); y.shape = (3,10)
        z = np.arange(30, dtype=np.complex128); z.shape = (3,10)
        d,w = {'x*y':x*y, 'x*z':x*z, 'y*z':y*z}, {}
        for k in list(d.keys()): w[k] = np.ones(d[k].shape)
        sol0 = {}
        for k in 'xyz': sol0[k] = eval(k) + .01
        ls = linsolve.LinProductSolver(d,sol0,w,sparse=self.sparse)
        ls.prm_order = {'x':0,'y':1,'z':2}
        sol = ls.solve()
        np.testing.assert_almost_equal(sol['x'], x, 2)
        np.testing.assert_almost_equal(sol['y'], y, 2)
        np.testing.assert_almost_equal(sol['z'], z, 2)

    def test_complex_array_NtimesNfreqs1_solve(self):
        x = np.arange(1, dtype=np.complex128); x.shape = (1,1)
        y = np.arange(1, dtype=np.complex128); y.shape = (1,1)
        z = np.arange(1, dtype=np.complex128); z.shape = (1,1)
        d,w = {'x*y':x*y, 'x*z':x*z, 'y*z':y*z}, {}
        for k in list(d.keys()): w[k] = np.ones(d[k].shape)
        sol0 = {}
        for k in 'xyz': sol0[k] = eval(k) + .01
        ls = linsolve.LinProductSolver(d,sol0,w,sparse=self.sparse)
        ls.prm_order = {'x':0,'y':1,'z':2}
        sol = ls.solve()
        np.testing.assert_almost_equal(sol['x'], x, 2)
        np.testing.assert_almost_equal(sol['y'], y, 2)
        np.testing.assert_almost_equal(sol['z'], z, 2)

    def test_sums_of_products(self):
        x = np.arange(1,31)*(1.0+1.0j); x.shape=(10,3) 
        y = np.arange(1,31)*(2.0-3.0j); y.shape=(10,3)
        z = np.arange(1,31)*(3.0-9.0j); z.shape=(10,3)
        w = np.arange(1,31)*(4.0+2.0j); w.shape=(10,3)
        x_,y_,z_,w_ = list(map(np.conjugate,(x,y,z,w)))
        expressions = ['x*y+z*w', '2*x_*y_+z*w-1.0j*z*w', '2*x*w', '1.0j*x + y*z', '-1*x*z+3*y*w*x+y', '2*w_', '2*x_ + 3*y - 4*z']
        data = {}
        for ex in expressions: data[ex] = eval(ex)
        currentSol = {'x':1.1*x, 'y': .9*y, 'z': 1.1*z, 'w':1.2*w}
        for i in range(5): # reducing iters prevents printing a bunch of "The exact solution is  x = 0" 
            testSolve = linsolve.LinProductSolver(data, currentSol,sparse=self.sparse)
            currentSol = testSolve.solve()
        for var in 'wxyz': 
            np.testing.assert_almost_equal(currentSol[var], eval(var), 4) 

    def test_eval(self):
        x = np.arange(1,31)*(1.0+1.0j); x.shape=(10,3) 
        y = np.arange(1,31)*(2.0-3.0j); y.shape=(10,3)
        z = np.arange(1,31)*(3.0-9.0j); z.shape=(10,3)
        w = np.arange(1,31)*(4.0+2.0j); w.shape=(10,3)
        x_,y_,z_,w_ = list(map(np.conjugate,(x,y,z,w)))
        expressions = ['x*y+z*w', '2*x_*y_+z*w-1.0j*z*w', '2*x*w', '1.0j*x + y*z', '-1*x*z+3*y*w*x+y', '2*w_', '2*x_ + 3*y - 4*z']
        data = {}
        for ex in expressions: data[ex] = eval(ex)
        currentSol = {'x':1.1*x, 'y': .9*y, 'z': 1.1*z, 'w':1.2*w}
        for i in range(5): # reducing iters prevents printing a bunch of "The exact solution is  x = 0" 
            testSolve = linsolve.LinProductSolver(data, currentSol,sparse=self.sparse)
            currentSol = testSolve.solve()
        for var in 'wxyz': 
            np.testing.assert_almost_equal(currentSol[var], eval(var), 4)
        result = testSolve.eval(currentSol)
        for eq in data:
            np.testing.assert_almost_equal(data[eq], result[eq], 4)

    def test_chisq(self):
        x = 1.
        d = {'x*y':1, '.5*x*y+.5*x*y':2, 'y':1}
        currentSol = {'x':2.3,'y':.9}
        for i in range(5): # reducing iters prevents printing a bunch of "The exact solution is  x = 0" 
            testSolve = linsolve.LinProductSolver(d, currentSol,sparse=self.sparse)
            currentSol = testSolve.solve()
        chisq = testSolve.chisq(currentSol)
        np.testing.assert_almost_equal(chisq, .5)

    def test_solve_iteratively(self):
        x = np.arange(1,31)*(1.0+1.0j); x.shape=(10,3) 
        y = np.arange(1,31)*(2.0-3.0j); y.shape=(10,3)
        z = np.arange(1,31)*(3.0-9.0j); z.shape=(10,3)
        w = np.arange(1,31)*(4.0+2.0j); w.shape=(10,3)
        x_,y_,z_,w_ = list(map(np.conjugate,(x,y,z,w)))
        expressions = ['x*y+z*w', '2*x_*y_+z*w-1.0j*z*w', '2*x*w', '1.0j*x + y*z', '-1*x*z+3*y*w*x+y', '2*w_', '2*x_ + 3*y - 4*z']
        data = {}
        for ex in expressions: data[ex] = eval(ex)
        currentSol = {'x':1.1*x, 'y': .9*y, 'z': 1.1*z, 'w':1.2*w}
        testSolve = linsolve.LinProductSolver(data, currentSol,sparse=self.sparse)
        meta, new_sol = testSolve.solve_iteratively()
        for var in 'wxyz': 
            np.testing.assert_almost_equal(new_sol[var], eval(var), 4)

    def test_solve_iteratively_dtype(self):
        x = np.arange(1,31)*(1.0+1.0j); x.shape=(10,3) 
        y = np.arange(1,31)*(2.0-3.0j); y.shape=(10,3)
        z = np.arange(1,31)*(3.0-9.0j); z.shape=(10,3)
        w = np.arange(1,31)*(4.0+2.0j); w.shape=(10,3)
        x_,y_,z_,w_ = list(map(np.conjugate,(x,y,z,w)))
        expressions = ['x*y+z*w', '2*x_*y_+z*w-1.0j*z*w', '2*x*w', '1.0j*x + y*z', '-1*x*z+3*y*w*x+y', '2*w_', '2*x_ + 3*y - 4*z']
        data = {}
        for dtype in (np.complex128, np.complex64):
            for ex in expressions: 
                data[ex] = eval(ex).astype(dtype)
            currentSol = {'x':1.1*x, 'y': .9*y, 'z': 1.1*z, 'w':1.2*w}
            currentSol = {k:v.astype(dtype) for k,v in currentSol.items()}
            testSolve = linsolve.LinProductSolver(data, currentSol,sparse=self.sparse)
            # some ridiculousness to avoid "The exact solution is  x = 0" prints
            save_stdout = sys.stdout
            sys.stdout = io.StringIO()
            meta, new_sol = testSolve.solve_iteratively(conv_crit=1e-7)
            sys.stdout = save_stdout
            for var in 'wxyz':
                assert new_sol[var].dtype == dtype
                np.testing.assert_almost_equal(new_sol[var], eval(var), 4)

    def test_degen_sol(self):
        # test how various solvers deal with degenerate solutions
        x,y,z = 1.+1j, 2.+2j, 3.+3j
        d,w = {'x*y_':x*y.conjugate(), 'x*z_':x*z.conjugate(), 'y*z_':y*z.conjugate()}, {}
        for k in list(d.keys()): w[k] = 1.
        sol0 = {}
        for k in 'xyz': sol0[k] = eval(k) + .01
        ls = linsolve.LinProductSolver(d,sol0,w,sparse=self.sparse)
        ls.prm_order = {'x':0,'y':1,'z':2}
        for mode in ('pinv', 'lsqr'):
            _, sol = ls.solve_iteratively(mode=mode)
            x,y,z = sol['x'], sol['y'], sol['z']
            np.testing.assert_almost_equal(x*y.conjugate(), d['x*y_'], 3)
            np.testing.assert_almost_equal(x*z.conjugate(), d['x*z_'], 3)
            np.testing.assert_almost_equal(y*z.conjugate(), d['y*z_'], 3)
        #self.assertRaises(np.linalg.LinAlgError, ls.solve_iteratively, mode='solve') # this fails for matrices where machine precision breaks degeneracies in system of equations

class TestLinProductSolverSparse(TestLinProductSolver):
    def setup(self):
        self.sparse=True
