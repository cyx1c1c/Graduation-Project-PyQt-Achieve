def testB():
    k = 0
    p = 1
    num = 0
    while p <= 20255202:
        p = 10 * k + 45
        x = str(p)[0]
        mark = 0
        for i in range(len(str(p))):
            if str(p)[i] == x:
                mark = 1
            else:
                mark = 0
                break
        if mark == 1:  # 各个数位上的数字都相同
            print(f"p:{p}\t", end=" ")
            print(f"k:{k}")
            num += p
        k += 1
    print(num)


def testC():
    n = int(input())
    num = 0
    for i in range(25):
        num += (n + i)
    print(num)


def testD():
    ls_lanqiao = ["L", "A", "N", "Q", "I", "A", "O"]
    h, w = map(int, input().split())
    sum = 0
    ls = []
    ls_sum = []
    for j in range(h):
        for i in range(w):
            ls.append(ls_lanqiao[(i + j) % len(ls_lanqiao)])
    for i in ls:
        if i == "A":
            sum += 1
    print(sum)


def testE():
    S = input()
    p = 0  # 头指针
    q = len(S) - 1  # 尾指针
    while p < q:
        if S[p] == "A":
            if S[q] == "A":  # 尾指针向左缩进
                q -= 1
            elif S[q] == "B":  # 删除所指的AB
                S = S[:p] + S[p + 1:q] + S[q + 1:]
                q = q - 2
        elif S[p] == "B":
            if S[q] == "A":  # 两边同时缩进
                p += 1
                q -= 1
            elif S[q] == "B":  # 头指针向右缩进
                p += 1
                continue
    print(len(S))


def testF():
    n = int(input())
    ls_a = list(input().split())
    ls_b = list(input().split())
    for i in range(n):  # 冒泡排序，从小到大
        for j in range(n - i - 1):
            if int(ls_a[j]) > int(ls_a[j + 1]):
                ls_a[j], ls_a[j + 1] = ls_a[j + 1], ls_a[j]
            if int(ls_b[j]) > int(ls_b[j + 1]):
                ls_b[j], ls_b[j + 1] = ls_b[j + 1], ls_b[j]
    ls_sum = []  # 用来存放操作数
    for i in range(n):  # 暴力穷举，对数列B进行多次赛马型排序
        ls = ls_b[i:] + ls_b[:i]
        sum = 0
        for i in range(n):
            if int(ls_a[i]) <= int(ls[i]):
                sum += 1
        ls_sum.append(sum)
    print(min(ls_sum))


def testG():
    n, k = map(int, input().split())
    ls_point = []  # 顶点表
    for i in range(n):
        ls_point.append(i + 1)
    ls_weight = list(input().split())  # 权重表
    ls_point_point = [[] for i in range(n)]  # 边表
    for i in range(1, n + 1):  # 生成n*n的矩阵，值均为0
        for j in range(1, n + 1):
            ls_point_point[i - 1].append(0)
    for i in range(n - 1):
        u, v = map(int, input().split())
        ls_point_point[u - 1][v - 1] = 1
    # print(ls_point_point)
    ls_sum_point = []  # 用来存放所能走到的结点数
    ls_sum_point.append(1)
    if k == 0:
        pass
    else:  # 总共能走2*k步
        for i in range(1, 2 * k + 1):
            DFS(i - 1, 1, n, ls_point_point, k, ls_sum_point)

    sum_G = 0  # 总价值
    for i in range(1, n + 1):
        if i in ls_sum_point:
            sum_G += int(ls_weight[i - 1])
    print(sum_G)


def DFS(step, u, n, ls_point_point, k, ls_sum_point):
    for v in range(u + 1, n + 1):
        if ls_point_point[u - 1][v - 1] == 1 and step == 0:
            ls_sum_point.append(v)
        elif ls_point_point[u - 1][v - 1] == 1 and step > 0:
            DFS(step - 1, v, n, ls_point_point, k, ls_sum_point)


def testH():
    S = list(input())
    vue_max = 0  # 用来存放最先且最大的字母的序号
    ls_s = []  # 存放结果
    for i in range(len(S)):
        if ord(S[i]) > ord(S[vue_max]):
            vue_max = i
    q = vue_max  # 设q为指针
    ls_s.append(S[q])
    while q <= len(S) - 1:
        vue_max_q = q + 1  # 用来存放q+1位置的最大字母的序号
        if vue_max_q >= len(S):
            break
        for i in range(q + 1, len(S)):
            if ord(S[i]) > ord(S[vue_max_q]):
                vue_max_q = i
        q = vue_max_q
        ls_s.append(S[q])
    result_s = ''
    for i in ls_s:
        result_s += i
    print(result_s)


if __name__ == '__main__':
    testH()