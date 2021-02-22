def foo(a=1, b="abc", **kwargs):
    print('a = ', a)
    print('b = ', b)
    print('kwargs = ', kwargs)
    print('---------------------------------------')


if __name__ == '__main__':
    foo(a=1, c=3)
