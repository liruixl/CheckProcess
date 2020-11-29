
import sys


class Solution(object):
    def nthUglyNumber(self, n, arr):
        res = 0
        points = [1]*len(arr)

        for _ in range(1, n + 1):

            num = min(arr[i]*points[i] for i in range(len(arr)))

            if num > n:
                break

            res += 1
            for i in range(len(arr)):
                if arr[i] * points[i] == num:
                    points[i] += 1
        return res


if __name__ == '__main__':
    s = Solution()
    n = 10
    m = 2
    arr = [3, 4]
    res = s.nthUglyNumber(n, arr)

    print(res)
