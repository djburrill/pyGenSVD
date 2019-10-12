'''
paige_gsvd.py

Description:
    Calculate GSVD of triangular matrix by Paige algorithm.

Author:
    Daniel Burrill

References:
    Paige, Christopher C., and Michael A. Saunders. "Towards a generalized singular value decomposition." SIAM Journal on Numerical Analysis 18.3 (1981): 398-405.

    Bai, Zhaojun, and James W. Demmel. "Computing the generalized singular value decomposition." SIAM Journal on Scientific Computing 14.6 (1993): 1464-1486.
'''

# Imports
import numpy as np
import numpy.linalg as nlg
import scipy.optimize as sco
import gsvd22

# Functions
def pgsvd(A, B, maxCycles=100, tol=0.001):
    '''
    Description:
        Compute generalize singular values of matrix A and B by Paige procedure.

    INPUT:
        A: [Numpy matrix, size(n,n)] Input matrix.
        B: [Numpy matrix, size(n,n)] Input matrix.
        maxCyles: [Integer] Stopping criteria for convergence of norm(A). Maximum number of cycles.
        tol: [Double] Stopping criteria for convergence of norm(A). Specifies tolerance of change in norm(A).

    OUTPUT:
        A: [Numpy matrix] C*R. C is diagonal.
        B: [Numpy matrix] S*R. S is diagonal.
        Ufinal: [Numpy matrix] Unitary matrix.
        Vfinal: [Numpy matrix] Unitary matrix.
        Qfinal: [Numpy matrix] Common unitary matrix.
    '''

    # Variables
    iterCount = 0
    rowsA = A.shape[0]
    colsA = A.shape[1]
    normAold = nlg.norm(A)
    normBold = nlg.norm(B)
    normAnew = normAold + 2*tol
    normBnew = normBold + 2*tol

    # Build Ufinal and Vfinal matrices
    Ufinal = np.asmatrix(np.eye(rowsA,colsA))
    Vfinal = np.asmatrix(np.eye(rowsA,colsA))
    Qfinal = np.asmatrix(np.eye(rowsA,colsA))

    # Convergence loop
    while (iterCount < maxCycles) and ((np.fabs(normAold-normAnew) >= tol) or (np.fabs(normBold-normBnew) >= tol)):
        # Determine triangularity
        tri = gsvd22.isTri(A)

        # Upper triangular
        if (tri == 0):
            for numRow in range(rowsA):
                for numCol in range(numRow+1,colsA):
                    # Build 2x2 matrices
                    A22 = np.matrix([[A[numRow,numRow],A[numRow,numCol]],[0.0,A[numCol,numCol]]])
                    B22 = np.matrix([[B[numRow,numRow],B[numRow,numCol]],[0.0,B[numCol,numCol]]])

                    # Compute GSVD of 2x2
                    A22,B22,U22,V22,Q22 = gsvd22.gsvd22(A22,B22)

                    # Build intermediate matrices
                    U = np.asmatrix(np.eye(rowsA,colsA))
                    V = np.asmatrix(np.eye(rowsA,colsA))
                    Q = np.asmatrix(np.eye(rowsA,colsA))

                    U[numRow,numRow] = U22[0,0]
                    U[numRow,numCol] = U22[0,1]
                    U[numCol,numRow] = U22[1,0]
                    U[numCol,numCol] = U22[1,1]

                    V[numRow,numRow] = V22[0,0]
                    V[numRow,numCol] = V22[0,1]
                    V[numCol,numRow] = V22[1,0]
                    V[numCol,numCol] = V22[1,1]

                    Q[numRow,numRow] = Q22[0,0]
                    Q[numRow,numCol] = Q22[0,1]
                    Q[numCol,numRow] = Q22[1,0]
                    Q[numCol,numCol] = Q22[1,1]

                    # Compute A_{k+1} and B_{k+1}
                    A = U.H*A*Q
                    B = V.H*B*Q
                    Ufinal = Ufinal*U
                    Vfinal = Vfinal*V
                    Qfinal = Qfinal*Q

        # Lower triangular
        elif (tri == 1):
            for numRow in range(rowsA):
                for numCol in range(numRow):
                    # Build 2x2 matrices
                    A22 = np.matrix([[A[numRow,numRow],0.0],[A[numRow,numCol],A[numCol,numCol]]])
                    B22 = np.matrix([[B[numRow,numRow],0.0],[B[numRow,numCol],B[numCol,numCol]]])

                    # Compute GSVD of 2x2
                    A22,B22,U22,V22,Q22 = gsvd22.gsvd22(A22,B22)

                    # Build intermediate matrices
                    U = np.asmatrix(np.eye(rowsA,colsA))
                    V = np.asmatrix(np.eye(rowsA,colsA))
                    Q = np.asmatrix(np.eye(rowsA,colsA))

                    U[numRow,numRow] = U22[0,0]
                    U[numRow,numCol] = U22[1,0]
                    U[numCol,numRow] = U22[0,1]
                    U[numCol,numCol] = U22[1,1]

                    V[numRow,numRow] = V22[0,0]
                    V[numRow,numCol] = V22[1,0]
                    V[numCol,numRow] = V22[0,1]
                    V[numCol,numCol] = V22[1,1]

                    Q[numRow,numRow] = Q22[0,0]
                    Q[numRow,numCol] = Q22[1,0]
                    Q[numCol,numRow] = Q22[0,1]
                    Q[numCol,numCol] = Q22[1,1]

                    # Compute A_{k+1} and B_{k+1}
                    A = U.H*A*Q
                    B = V.H*B*Q
                    Ufinal = Ufinal*U
                    Vfinal = Vfinal*V
                    Qfinal = Qfinal*Q

        # Compute norms
        normAold = normAnew
        normAnew = nlg.norm(A)
        normBold = normBnew
        normBnew = nlg.norm(B)
        iterCount = iterCount+1

    return A,B,Ufinal,Vfinal,Qfinal

if (__name__ == '__main__'):
    '''
    # 2x2 LT
    A = np.matrix([[2.0,0.0],[1.0,1.0]])
    B = np.matrix([[1.0,0.0],[3.0,1.0]])

    print '2x2 LT'
    print A
    print B

    A1,B1,U,V,Q = pgsvd(A,B)

    print U*A1*Q.H
    print V*B1*Q.H

    # 2x2 UT
    A = np.matrix([[2.0,1.0],[0.0,1.0]])
    B = np.matrix([[1.0,5.0],[0.0,1.0]])

    print '\n2x2 UT'
    print A
    print B

    A1,B1,U,V,Q = pgsvd(A,B)

    print U*A1*Q.H
    print V*B1*Q.H

    # 3x3 LT
    A = np.matrix([[2.0,0.0,0.0],[1.0,1.0,0.0],[3.0,4.0,5.0]])
    B = np.matrix([[1.0,0.0,0.0],[3.0,1.0,0.0],[2.0,4.0,5.0]])

    print '\n3x3 LT'
    print A
    print B

    A1,B1,U,V,Q = pgsvd(A,B)

    print U*A1*Q.H
    print V*B1*Q.H

    # 3x3 UT
    A = np.matrix([[2.0,3.0,4.0],[0.0,1.0,2.0],[0.0,0.0,-5.0]])
    B = np.matrix([[1.0,2.0,-1.0],[0.0,1.0,2.0],[0.0,0.0,5.0]])

    print '\n3x3 UT'
    print A
    print B

    A1,B1,U,V,Q = pgsvd(A,B)

    print U*A1*Q.H
    print V*B1*Q.H
    '''
