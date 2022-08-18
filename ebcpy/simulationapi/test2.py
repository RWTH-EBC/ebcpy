import multiprocessing as mp

class MyClass:
    _items_to_drop = [
        'pool',
    ]
    def __init__(self, n_cpu):
        self.n_cpu = n_cpu
        self.pool = mp.Pool(processes=self.n_cpu)
        self.my_first_method()

    def __getstate__(self):
        """Overwrite magic method to allow pickling the api object"""
        self_dict = self.__dict__.copy()
        for item in self._items_to_drop:
            del self_dict[item]
        #return deepcopy(self_dict)
        return self_dict

    def __setstate__(self, state):
        """Overwrite magic method to allow pickling the api object"""
        self.__dict__.update(state)

    def my_first_method(self):
        self.pool.map(
            self.my_second_method,
            [True for _ in range(self.n_cpu)]
        )
        print('done')


    def my_second_method(self, input):
        if input:
            print('Yes')


def main(n_cpu):
    obj = MyClass(n_cpu=n_cpu)
    print('end')


if __name__ == '__main__':
    print('start')
    main(
        n_cpu=2,
        )
    print('end_end')

    
