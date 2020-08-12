


import sys



if __name__ == '__main__':
    m = 2
    n = 5

    res = []
    one = []

    def dfs(idx, one, res):

        if len(one) == m:
            a = one.copy()
            res.append(a)

        for i in range(idx, n + 1):
            one.append(i)
            dfs(i + 1, one, res)
            one.pop(-1)

    dfs(1, one, res)

    print(res)

    for pp in res:
        print('{0} {1}'.format(pp[0], pp[1]))