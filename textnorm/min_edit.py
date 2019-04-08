
def sub_cost(a, b):
    """
    Get substitution cost between strings a and b.

    Args:
        a (str)
        b (str)

    Returns:
        int: Equals two if a!=b, else zero.
    """
    if a == b:
        return 0
    else:
        return 2


def min_edit_distance(source, target):
    """
    Compute minimum edit distance between strings
    'source' and 'target'. Use dynamic programming.

    Args:
        source (str)
        target (str)

    Returns:
        min_distance (str)
    """
    n = len(source)
    m = len(target)

    # Create empty distance matrix
    D = list()
    for _ in range(n+1):
        D.append([0]*(m+1))

    # Initialize: the zeroth row and column is the distance
    # from the empty string
    D[0][0] = 0
    for i in range(1, n+1):
        D[i][0] = D[i-1][0] + 1
    for j in range(1, m+1):
        D[0][j] = D[0][j-1] + 1

    # Recurrence relation
    for i in range(1, n+1):
        for j in range(1, m+1):
            _del = D[i-1][j] + 1
            _sub = D[i-1][j-1] + sub_cost(source[i-1], target[j-1])
            _ins = D[i][j-1] + 1
            D[i][j] = min(_del, _sub, _ins) 

    return D[n][m]


def print_grid(D):
    for line in D:
        print(line)
