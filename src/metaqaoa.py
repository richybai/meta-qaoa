import numpy as np
import networkx as nx
from mindquantum import Circuit, H, ZZ, RX, BarrierGate
from mindspore import Model


def gene_random_instance(num_nodes):
    """
    函数生成随机图
    输入是 图的结点个数 num_nodes, 文章里是n
    输出是图
    """
    k = np.random.randint(3, num_nodes)
    # 两个顶点连接的概率 p = k / n, 文中没说生成 k-regular graph.
    p = k / num_nodes
    g = nx.Graph()
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if np.random.random() < p:
                nx.add_path(g, [i, j])
    return g


def gene_Hc_circuit(G, param):
    """
    生成HC2对应的线路，也就是ZZ门作用到对应的qubits上
    input: G maxcut的图, param 这一层Hc对应的参数
    output: HC2 circuit
    """
    hc_circuit = Circuit()
    for e in G.edges:
        hc_circuit += ZZ(param).on(e)
    hc_circuit += BarrierGate(False)
    return hc_circuit


def gene_Hm_circuit(G, param):
    """
    生成HM对应的线路，也就是RX门作用到对应的qubits上
    input: G maxcut的图, param 这一层Hc对应的参数
    output: HM circuit
    """
    hm_circuit = Circuit()
    for v in G.nodes:
        hm_circuit += RX(param).on(v)
    hm_circuit += BarrierGate(False)
    return hm_circuit


def gene_qaoa_ansatz(G, P):
    """
    使用 hc hm 生成 qaoa的ansatz
    input: G, 对应的图  P: hc hm 重复的次数
    output: ansatz circuit
    """
    ansatz = Circuit()
    for i in G.nodes:
        ansatz += H(i)
    for i in range(P):
        ansatz += gene_Hc_circuit(G, f"g{i}")
        ansatz += gene_Hm_circuit(G, f"m{i}")
    return ansatz


if __name__ == "__main__":
    # 测试生成图
    g = gene_random_instance(4)
    print(g)
    print(g.nodes)
    print(g.edges)
    # 测试hc hm and ansatz
    # hc = gene_Hc_circuit(g, 'g0')
    # print(hc)
    # hm = gene_Hm_circuit(g, 'm0')
    # print(hm)
    ansatz = gene_qaoa_ansatz(g, 3)
    print(ansatz)