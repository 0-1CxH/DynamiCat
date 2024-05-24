import time

from dynamicat.utils.cpu_parallel import mproc_map, threads_pool_map


if __name__ == '__main__':
    class F:
        @classmethod
        def f(cls, x):
            if x % 5 == 0:
                time.sleep(0.5)
            if x % 3 == 0:
                time.sleep(0.3)
            return x*2

    #
    print(mproc_map(F.f, list(range(15)), 3, True, 2))
    print(mproc_map(F.f, list(range(15)), 3, False, 2))

    print(threads_pool_map(F.f, list(range(30)), True, 3))
    print(threads_pool_map(F.f, list(range(30)), False, 3))


