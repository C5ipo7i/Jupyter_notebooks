# Permuted multiples

def permuted():
    x = 125874
    while True:
        digits = set([int(d) for d in str(x)])
        print('digits',digits)
        for i in range(2,7):
            y = x*i
            # print('y',y)
            temp = set([int(d) for d in str(y)])
            # print('temp',temp)
            # print('diff',temp ^ digits)
            if (temp ^ digits):
                break
            elif i == 6:
                return x
        x += 1


if __name__ == "__main__":
    print(permuted())
