import sys


def solve(size, arr):

    idxs = arr.copy()
    for i in range(size):

        while True:
            idx = idxs[arr[i]]
            if idx == i or arr[i] == arr[idx]:
                break

            temp = arr[i]
            arr[i] = arr[idx]
            arr[idx] = temp

    return arr


if __name__ == '__main__':
    size = 8
    arr = [7,6,5,4,0,1,2,3]

    solve(size, arr)
    print(arr)