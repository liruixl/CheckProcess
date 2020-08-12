from collections import defaultdict

def max_overlap(nums1, nums2):
    index_num = defaultdict(int)
    res = 0
    tmp = 0
    for i in nums1:
        index_num[i] += 1
    for j in nums2:
        index_num[j] -= 1
    nums = sorted(index_num.items(), key = lambda x: x[0])
    for i in nums:
        tmp += i[1]
        res = max(res, tmp)
    return res


if __name__ == '__main__':
    import sys
    num = int(sys.stdin.readline().strip())
    num1 = []
    num2 = []
    for i in range(num):
        tmp = list(map(int, sys.stdin.readline().strip().split()))
        num1.append(tmp[0])
        num2.append(tmp[1])
    print(max_overlap(num1, num2))