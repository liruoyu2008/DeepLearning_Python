def foo(a, b="abc", **kwargs):
    print('a = ', a)
    print('b = ', b)
    print('kwargs = ', kwargs)
    print('---------------------------------------')


if __name__ == '__main__':
    foo()
