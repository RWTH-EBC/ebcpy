from multiprocessing import Pool


def f(x):
    return x*x


def mp_mapping():
    with Pool(4) as p:
        return p.map(f, range(4))


if __name__ == '__main__':
    res = mp_mapping()
    print(res)

