def foo(a=1, b="abc", **kwargs):
    print('a = ', a)
    print('b = ', b)
    print('kwargs = ', kwargs)
    print('---------------------------------------')


def create_labels(filepath:str):
    """根据给定的文件路径生成标签文件

    Args:
        filepath (str): 文件完整路径，包含文件名
    """    
    f = open(filepath, 'w')
    list1 = ['1\n' for i in range(0, 100)]
    list2 = ['2\n' for i in range(0, 100)]
    list = list1+list2
    f.writelines(list)
    f.flush()
    f.close()
    print('success!')


if __name__ == '__main__':
    create_labels(r'C:\Users\Ryu\Desktop\train\labels.txt')
