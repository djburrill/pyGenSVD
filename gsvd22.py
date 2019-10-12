'''
gsvd22.py

Description:
    Calculate GSVD of 2x2 triangular matrix by Paige algorithm.

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
import adj
import kogbetliantz as kbl

# Functions
def buildRotMat(angle):
    '''
    Description:
        Build 2x2 rotation matrix.

    INPUT:
        angle: [Double] Rotation angle.

    OUTPUT:
        outMat: [Numpy matrix] Rotation matrix.
    '''

    outMat = np.asmatrix(np.zeros((2,2)))

    outMat[0,0] = np.cos(angle)
    outMat[0,1] = np.sin(angle)
    outMat[1,0] = -np.sin(angle)
    outMat[1,1] = np.cos(angle)

    return outMat

def upQ(xIn,M):
    '''
    Description:
        Determine angle for Q to zero out upper triangular portion of M.

    INPUT:
        xIn: [Double] Guess angle.
        M: [Numpy matrix] Input matrix.

    OUTPUT:
        xOut: [Double] Value of upper right element of M after rotation by xIn.
    '''

    xOut = 0.0

    xOut = M[0,0]*np.sin(xIn)+M[0,1]*np.cos(xIn)

    return xOut

def lowQ(xIn,M):
    '''
    Description:
        Determine angle for Q to zero out lower triangular portion of M.

    INPUT:
        xIn: [Double] Guess angle.
        M: [Numpy matrix] Input matrix.

    OUTPUT:
        xOut: [Double] Value of lower left element of M after rotation by xIn.
    '''

    xOut = 0.0

    xOut = M[1,0]*np.cos(xIn)-M[1,1]*np.sin(xIn)

    return xOut

def upTri(inMat,tol=1e-10):
    '''
    Description:
        Determine if matrix is upper triangular by element-wise check.

    INPUT:
        inMat: [Numpy matrix] Input matrix.
        tol: [Double] Tolerance threshold for zero evaluation.

    OUTPUT:
        [Boolean]: True if upper triangular, False otherwise.
    '''

    # Variables
    rows = inMat.shape[0]                   # Number of rows in inMat
    cols = inMat.shape[1]                   # Number of cols in inMat

    # Check lower triangular elements
    for index1 in range(rows):
        for index2 in range(0,index1):
            if ( np.fabs(inMat[index1,index2])-tol >= 0.0):
                return False

    return True

def lowTri(inMat,tol=1e-10):
    '''
    Description:
        Determine if matrix is lower triangular by element-wise check.

    INPUT:
        inMat: [Numpy matrix] Input matrix.
        tol: [Double] Tolerance threshold for zero evaluation.

    OUTPUT:
        [Boolean]: True if lower triangular, False otherwise.
    '''

    # Variables
    rows = inMat.shape[0]                   # Number of rows in inMat
    cols = inMat.shape[1]                   # Number of cols in inMat

    # Check upper triangular elements
    for index1 in range(rows):
        for index2 in range(index1+1,cols):
            if ( np.fabs(inMat[index1,index2])-tol >= 0.0):
                return False

    return True

def nonZero(inMat,tol=1e-10):
    '''
    Description:
        Determine if matrix is nonzero by element-wise check.

    INPUT:
        inMat: [Numpy matrix] Input matrix.
        tol: [Double] Tolerance threshold for zero evaluation.

    OUTPUT:
        [Boolean]: True if non-zero matrix, False otherwise.

    Returns Boolean
    '''

    # Variables
    rows = inMat.shape[0]                   # Number of rows in inMat
    cols = inMat.shape[1]                   # Number of cols in inMat

    # Check elements
    for index1 in range(rows):
        for index2 in range(cols):
            if ( np.fabs(inMat[index1,index2])-tol >= 0.0):
                return False

    return True

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

def computeAngles(xIn,A,U,tri):
    '''
    Description:
        Determine phi to diagonalize A.

    INPUT:
        xIn: [Double] Guess angle.
        A: [Numpy matrix] Input matrix.
        U: [Numpy matrix] Rotation matrix.
        tri: [Integer] Triangularity of A. Must be triangular. Uses same convention as isTri().

    OUTPUT:
        xOut: [Double] Value of diagonal element to diagonalize A.
    '''

    # Variables
    xOut = 0.0

    # Upper triangular
    if (tri == 0):
        xOut = U[0,0]*np.sin(xIn)*A[0,0] + (A[0,1]*U[0,0]+A[1,1]*U[0,1])*np.cos(xIn)
    # Lower triangular
    else:
        xOut = -U[0,1]*np.cos(xIn)*A[1,1] + (A[0,0]*U[0,0]+A[1,0]*U[0,1])*np.sin(xIn)

    return xOut

def gsvd22(A,B):
    '''
    Description:
        Compute GSVD of 2x2 matrix.

    INPUT:
        A: [Numpy matrix, size(2,2)] Input matrix.
        B: [Numpy matrix, size(2,2)] Input matrix.

    OUTPUT:
        A1: [Numpy matrix, size(2,2)] C*R. C is diagonal.
        B1: [Numpy matrix, size(2,2)] S*R. S is diagonal.
        U: [Numpy matrix, size(2,2)] Unitary matrix.
        Vh.H: [Numpy matrix, size(2,2)] Unitary matrix.
        Q: [Numpy matrix, size(2,2)] Common unitary matrix.
    '''

    # Variables
    Q = np.asmatrix(np.zeros((2,2)))
    angle = 0.0

    # Compute C
    C = A*adj.adjugate(B)

    # SVD of C
    S,U,V = kbl.kbl(C)
    Vh = V.H

    # Compute G and H
    G = U.H*A
    H = Vh*B

    # Upper triangular
    if (upTri(A) and upTri(B)):
        # Check if A is nonzero
        if (nonZero(A)):
            # Compute angle
            angle = sco.fsolve(upQ,angle,args=(G))

            # Build Q
            Q = buildRotMat(angle)
        # If A is zero then work with B
        else:
            # Compute angle
            angle = sco.fsolve(upQ,angle,args=(H))

            # Build Q
            Q = buildRotMat(angle)

        # Compute matrices
        A1 = G*Q
        B1 = H*Q
        A1[0,1] = 0.0
        B1[0,1] = 0.0

    # Lower triangular
    else:
        # Check if A is nonsingular
        if (nlg.det(A) != 0.0):
            # Compute angle
            angle = sco.fsolve(lowQ,angle,args=(G))

            # Build Q
            Q = buildRotMat(angle)
        # If A is zero then work with B
        else:
            # Compute angle
            angle = sco.fsolve(lowQ,angle,args=(H))

            # Build Q
            Q = buildRotMat(angle)

        # Compute matrices
        A1 = G*Q
        B1 = H*Q
        A1[1,0] = 0.0
        B1[1,0] = 0.0

    return A1,B1,U,Vh.H,Q

# Main
if (__name__ == '__main__'):
    '''
    # Test cases

    # Variables
    A = np.matrix([[2.0,0.0],[1.0,1.0]])
    B = np.matrix([[1.0,0.0],[3.0,1.0]])

    print 'A:'
    print A
    print 'B:'
    print B

    A1,B1,U,V,Q = gsvd22(A,B)

    print 'A1:'
    print A1
    print 'B1:'
    print B1

    print '==========='

    print 'U:'
    print U
    print 'V:'
    print V
    print 'Q:'
    print Q

    print '==========='

    b1 = 1.0/np.sqrt(1.0+(A1[0,0]/B1[0,0])*(A1[0,0]/B1[0,0]))
    a1 = np.sqrt(1-b1*b1)

    b2 = 1.0/np.sqrt(1.0+(A1[1,1]/B1[1,1])*(A1[1,1]/B1[1,1]))
    a2 = np.sqrt(1-b2*b2)

    C = np.matrix([[a1,0.0],[0.0,a2]])
    S = np.matrix([[b1,0.0],[0.0,b2]])

    print 'C:'
    print C
    print 'S:'
    print S

    print '==========='

    r00 = A1[0,0]/a1
    r01 = A1[0,1]/a1
    r11 = A1[1,1]/a2

    R = np.matrix([[r00,r01],[0.0,r11]])

    print 'R:'
    print R

    print 'X:'
    print Q*R.I
    #print np.dot(Q,nlg.inv(R))

    print 'Checks:'
    print C*R
    print S*R
    #print np.dot(C,R)
    #print np.dot(S,R)

    #print np.dot()
    print A1[0,1]/B1[0,1]

    # Convert to matrices
    A = np.asmatrix(A)
    B = np.asmatrix(B)
    A1 = np.asmatrix(A1)
    B1 = np.asmatrix(B1)
    U = np.asmatrix(U)
    V = np.asmatrix(V)
    Q = np.asmatrix(Q)
    C = np.asmatrix(C)
    S = np.asmatrix(S)
    R = np.asmatrix(R)

    print U*C*R*Q.H
    print V*S*R*Q.H
    '''
