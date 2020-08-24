
import sys




if __name__ == '__main__':

    kmap = {')': '(', ']': '['}

    kuohaos = ')(][][)('
    ans = 0

    st = []

    for i in range(len(kuohaos)):

        c = kuohaos[i]
        if c in kmap.keys():
            match = False

            while len(st) > 0:
                if st[-1] == kmap[c]:
                    st.pop(-1)
                    match = True
                    break
                else:
                    ans += 1
                    st.pop()

            if not match:
                ans += 1
        else:
            st.append(c)

    print(ans)

