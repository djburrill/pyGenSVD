'''
kogbetliantz.py

Description:
    Calculate SVD of triangular matrix by Kogbetliantz algorithm.

Author:
    Daniel Burrill

References:
    Kogbetliantz, E. G. "Solution of linear equations by diagonalization of coefficients matrix." Quarterly of Applied Mathematics 13.2 (1955): 123-132.

    Bai, Zhaojun, and James W. Demmel. "Computing the generalized singular value decomposition." SIAM Journal on Scientific Computing 14.6 (1993): 1464-1486.
'''

# Imports
import numpy as np
import numpy.linalg as nlg
import scipy.optimize as sco

# Functions
def isTri(inMat,tol=1.0e-11):
    '''
    Description:
        Determine if a matrix is upper (0), lower (1) triangular, or neither (-1).

    INPUT:
        inMat: [Numpy matrix] Input matrix.
        tol: [Double] Tolerance threshold for zero evaluation.

    OUTPUT:
        outVal: [Integer] Specify triangularity case:
            -1: Neither upper nor lower triangular
            0: Upper triangular
            1: Lower triangular
    '''

    # Variables
    outVal = -1
    rowsIn = inMat.shape[0]
    colsIn = inMat.shape[1]

    # Get triangular parts of matrix
    inMat_Up = np.absolute(np.triu(inMat,1))
    inMat_Low = np.absolute(np.tril(inMat,-1))

    # Check for lower triangular
    if (np.any(inMat_Up >= tol)):
        outVal = 0      # Upper triangular
    else:
        outVal = 1      # Lower triangular

    # Check for upper triangular
    if (np.any(inMat_Low >= tol)):
        if (outVal == 0):
            outVal = -1    # Neither
    else:
        if (outVal == 1):
            outVal = -1    # Neither

    return outVal

def computeAngles(xIn,A,tri):
    '''
    Description:
        Determine phi and psi to diagonalize A.

    INPUT:
        xIn: [Python list len(2), Double] List of angles to evaluate rotations.
        A: [Numpy matrix] Input matrix.
        tri: [Integer] Triangularity of A. Must be triangular. Uses same convention as isTri().

    OUTPUT:
        xOut: [Python list len(2), Double] List of computed angles.
    '''

    # Variables
    xOut = [0.0,0.0]

    # Upper triangular
    if (tri == 0):
        xOut[0] = A[0,0]*np.cos(xIn[0])*np.sin(xIn[1]) + (A[0,1]*np.cos(xIn[0])-A[1,1]*np.sin(xIn[0]))*np.cos(xIn[1])
        xOut[1] = A[0,0]*np.sin(xIn[0])*np.cos(xIn[1]) - (A[0,1]*np.sin(xIn[0])+A[1,1]*np.cos(xIn[0]))*np.sin(xIn[1])
    # Lower triangular
    else:
        xOut[0] = (A[0,0]*np.cos(xIn[0])-A[1,0]*np.sin(xIn[0]))*np.sin(xIn[1]) - A[1,1]*np.sin(xIn[0])*np.cos(xIn[1])
        xOut[1] = (A[0,0]*np.sin(xIn[0])+A[1,0]*np.cos(xIn[0]))*np.cos(xIn[1]) - A[1,1]*np.cos(xIn[0])*np.sin(xIn[1])

    return xOut

def positivize_diag(inMat):
    '''
    Decompose diagonal matrix with negative elements into a diagonal matrix with positive values and a permutation matrix.
    '''

    # Variables
    rowsIn = inMat.shape[0]
    colsIn = inMat.shape[1]
    P = np.asmatrix(np.eye(rowsIn,colsIn))
    diag = np.diag(inMat)

    # Get positions of negative elements
    pos = [index for index,val in enumerate(diag) if (val < 0.0)]

    # Place negatives in permutation matrix and remove from inMat
    for index in pos:
        P[index,index] = -P[index,index]
        inMat[index,index] = -inMat[index,index]

    return A,P

def kbl(A, maxCycles=100, tol=0.001):
    '''
    Description:
        Compute singular values of matrix A by Kogbetliantz procedure. Note that the algorithm finds the singular values of A in-place.

        A = U*Sigma*V.H

    INPUT:
        A: [Numpy matrix] Input matrix.
        maxCyles: [Integer] Stopping criteria for convergence of norm(A). Maximum number of cycles.
        tol: [Double] Stopping criteria for convergence of norm(A). Specifies tolerance of change in norm(A).

    OUTPUT:
        A: [Numpy matrix] Diagonal matrix of singular values. Normally called Sigma.
        Ufinal: [Numpy matrix] Unitary matrix U.
        Vfinal: [Numpy matrix] Unitary matrix V.
    '''

    # Variables
    iterCount = 0
    rowsA = A.shape[0]
    colsA = A.shape[1]
    normAold = nlg.norm(A)
    normAnew = normAold + 2*tol

    # Build Ufinal and Vfinal matrices
    Ufinal = np.asmatrix(np.eye(rowsA,colsA))
    Vfinal = np.asmatrix(np.eye(rowsA,colsA))

    # Convergence loop
    while (iterCount < maxCycles) and (np.fabs(normAold-normAnew) >= tol):
        # Determine triangularity of A
        triA = isTri(A)

        # Upper triangular A
        if (triA == 0):
            for numRow in range(rowsA):
                for numCol in range(numRow+1,colsA):
                    # Intialize angles
                    angles = [0.0,0.0]

                    # Build temporary matrix
                    tempA = np.matrix([[A[numRow,numRow],A[numRow,numCol]],[0.0,A[numCol,numCol]]])

                    # Compute angles
                    angles = sco.fsolve(computeAngles,angles,args=(tempA,0))

                    # Build U and V matrices
                    U = np.asmatrix(np.eye(rowsA,colsA))
                    V = np.asmatrix(np.eye(rowsA,colsA))

                    U[numRow,numRow] = np.cos(angles[0])
                    U[numRow,numCol] = np.sin(angles[0])
                    U[numCol,numRow] = -np.sin(angles[0])
                    U[numCol,numCol] = np.cos(angles[0])

                    V[numRow,numRow] = np.cos(angles[1])
                    V[numRow,numCol] = np.sin(angles[1])
                    V[numCol,numRow] = -np.sin(angles[1])
                    V[numCol,numCol] = np.cos(angles[1])

                    # Compute A_{k+1} and update Ufinal and Vfinal
                    A = U.H*A*V
                    Ufinal = Ufinal*U
                    Vfinal = Vfinal*V

        # Lower triangular
        elif (triA == 1):
            for numRow in range(rowsA):
                for numCol in range(numRow):
                    # Intialize angles
                    angles = [0.0,0.0]

                    # Build temporary matrix
                    tempA = np.matrix([[A[numCol,numCol],0.0],[A[numRow,numCol],A[numRow,numRow]]])

                    # Compute angles
                    angles = sco.fsolve(computeAngles,angles,args=(tempA,1))

                    # Build U and V matrices
                    U = np.asmatrix(np.eye(rowsA,colsA))
                    V = np.asmatrix(np.eye(rowsA,colsA))

                    U[numCol,numCol] = np.cos(angles[0])
                    U[numCol,numRow] = np.sin(angles[0])
                    U[numRow,numCol] = -np.sin(angles[0])
                    U[numRow,numRow] = np.cos(angles[0])

                    V[numCol,numCol] = np.cos(angles[1])
                    V[numRow,numCol] = -np.sin(angles[1])
                    V[numCol,numRow] = np.sin(angles[1])
                    V[numRow,numRow] = np.cos(angles[1])

                    # Compute A_{k+1} and update Ufinal and Vfinal
                    A = U.H*A*V
                    Ufinal = Ufinal*U
                    Vfinal = Vfinal*V

        # Compute norm
        normAold = normAnew
        normAnew = nlg.norm(A)
        iterCount = iterCount+1

    # Make sure singular values are all positive
    diag = np.diag(A)
    # If there are any negative elements
    if ((diag < 0.0).any()):
        A,P = positivize_diag(A)
        Ufinal = Ufinal*P

    # Sort matrices
    '''
    sv = np.diag(A)
    sv_sort = np.argsort(sv)[::-1]
    sv = np.sort(sv)[::-1]

    # Rebuild A as diagonal
    A = np.zeros((rowsA,colsA))
    np.fill_diagonal(A,sv)

    Ufinal[:,range(len(sv))] = Ufinal[:,sv_sort]
    Vfinal[:,range(len(sv))] = Vfinal[:,sv_sort]
    '''

    return A,Ufinal,Vfinal

if (__name__ == '__main__'):
    '''
    # Test cases

    # Variables
    A = np.matrix([[1.0,2.0,3.0],[0.0,4.0,5.0],[0.0,0.0,6.0]])
    Atest = np.matrix([[1.0,2.0],[0.0,3.0]])
    Atest2 = np.matrix([[1.0,0.0],[1.0,1.0]])
    AtestA = np.matrix([[2.0,0.0],[1.0,1.0]])
    AtestB = np.matrix([[1.0,0.0],[3.0,1.0]])
    AtestC = AtestA*AtestB.I
    #AtestC = np.dot(AtestA,nlg.inv(AtestB))

    print AtestC

    # Test isTri
    upper = np.array([[1.0,2.0,3.0],[0.0,4.0,5.0],[0.0,0.0,6.0]])
    lower = np.array([[1.0,0.0,0.0],[2.0,3.0,0.0],[4.0,5.0,6.0]])
    neither1 = np.array([[1.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,5.0]])
    neither2 = np.array([[1.0,1.0,0.0],[1.0,3.0,0.0],[0.0,5.0,0.0]])

    print isTri(upper)
    print isTri(lower)
    print isTri(neither1)
    print isTri(neither2)

    U,s,V = nlg.svd(AtestC)
    print 'Numpy SVD'
    print s
    print U
    print V
    print ''

    #U1,s1,V1 = nlg.svd(Atest)

    #print s1
    #print U1
    #print V1

    #  Test KBL procedure
    Amat,Uk,Vk = kbl(AtestC,tol=0.00001)

    print 'KBL SVD'
    print Amat
    print Uk
    print Vk.H
    '''
