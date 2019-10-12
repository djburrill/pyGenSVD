# Adjugate test

# Imports
import numpy as np
import numpy.linalg as nlg

# Functions
def subMat(inMat,rowRem,colRem):
    '''
    Return submatrix of inMat with rowRem and colRem removed
    '''

    # Variables
    rows = inMat.shape[0]                               # Number of rows in inMat
    cols = inMat.shape[1]                               # Number of cols in inMat
    outMat = np.asmatrix(np.ones((rows-1,cols-1)))      # Initialize outMat
    rowList = range(rows)                               # Create list of rows w/o rowRem
    rowList.remove(rowRem)
    colList = range(cols)                               # Create list of cols w/o colRem
    colList.remove(colRem)

    # Populate outMat
    for index1,rowIndex in enumerate(rowList):
        for index2, colIndex in enumerate(colList):
            outMat[index1,index2] = inMat[rowIndex,colIndex]

    return outMat

def adjugate(inMat):
    '''
    Compute adjugate of inMat by cofactor method.
    '''

    # Variables
    rows = inMat.shape[0]
    cols = inMat.shape[1]
    C = np.asmatrix(np.eye(rows,cols))

    for index1 in range(rows):
        for index2 in range(cols):
            # Build sub matrix
            subMatrix = subMat(inMat,index1,index2)

            # Even
            if ((index1+index2)%2 == 0):
                C[index1,index2] = nlg.det(subMatrix)
            # Odd
            else:
                C[index1,index2] = -nlg.det(subMatrix)

    return C.H

# Main
if (__name__ == '__main__'):
    pass
    '''
    # Variables
    A = np.array([[-3,2,-5],[-1,0,-2],[3,-4,1]])

    Adj = adjugate(A)

    print Adj
    '''
