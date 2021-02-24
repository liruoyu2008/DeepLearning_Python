def foo(a=1, b="abc", **kwargs):
    print('a = ', a)
    print('b = ', b)
    print('kwargs = ', kwargs)
    print('---------------------------------------')


if __name__ == '__main__':
    foo()
    a = ['1','2', '3']
    b="1212132443545t232323"
    c = b.title()
    print(c)
