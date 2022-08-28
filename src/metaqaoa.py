from re import T
import numpy as np
import networkx as nx
from mindquantum import Circuit, H, ZZ, RX, BarrierGate
from mindquantum import Hamiltonian, QubitOperator, Simulator, MQEncoderOnlyOps, MQAnsatzOnlyOps, MQLayer
from mindspore import nn, ops, Tensor
import mindspore as ms
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


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
    生成HC2对应的线路, 也就是ZZ门作用到对应的qubits上
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
    生成HM对应的线路, 也就是RX门作用到对应的qubits上
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


def gene_ham(G):
    ham = QubitOperator()
    for i in G.edges:
        ham += QubitOperator(f'Z{i[0]} Z{i[1]}')  # 生成哈密顿量Hc
    return Hamiltonian(ham)


def gene_qaoa_layers(G, P):
    """
    使用mindquantum framework 生成 QNN的层, 输出是期望值, 因为不需要在这里面更新参数，所以使用MQEncoderOnlyOps
    """
    ansatz = gene_qaoa_ansatz(G, P)
    ansatz.as_encoder()

    ham = gene_ham(G)

    sim = Simulator('projectq', ansatz.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, ansatz)
    net = MQEncoderOnlyOps(grad_ops)
    # net = MQLayer(grad_ops)
    return net

###################### 把QNN当成样本传给网络，造成报错TypeError: 'self.g' should be initialized as a 'Parameter' type in the '__init__' function, but got 'None' with type 'NoneType.

# class MetaQAOA(nn.Cell):
#     """
#     生成 MetaQAOA网络, 把LSTM看成主体结构, QNN看成是输入的数据
#     *** 把参数看成是seq_len为P, 对应到unitary每一次执行，最后输出时候会把input_size=2这一维度消掉，那么只有把seq_len设置为2P, 把输入拉直***
#     *** 因为y是多余的输入，把y当成c直接输入，正好和h凑在一起，且都是0 ***
#     """
#     def __init__(self, T):
#         """
#         T times of quantum classical interaction 
#         """
#         super(MetaQAOA, self).__init__()
#         # 初始化LSTM三个参数为input_size, hidden_size (int), num_layers (int)
#         self.lstm = nn.LSTM(1, 1, 1, has_bias=True, batch_first=True, bidirectional=False)
#         self.T = T

#     def construct(self, theta, h, y, QNN): # 
#         """
#         本来想实现batch的，但是写循环之后报错改不过来，在construct里面写batch=1的特例
#         input: 
#         1. theta是QNN的参数，长度是2P
#         2. y对应到的是LSTM的c 是QNN输出的期望，reshape成(1, 1, 1) 
#         3. h是hidden state, shape 是 (1, 1, 1)
#         4. QNN 是QAOA的线路，单个
#         output:
#         1. theta_list [T, 2*P]
#         2. y_list [T, (1)] 用于计算loss        
#         """
#         theta_list = []
#         y_list = []
#         for i in range(self.T):
#             theta, (h, _) = self.lstm(theta, (h, y))
#             y = QNN(theta.reshape(1, -1))
#             y_list.append(y.squeeze())
#             y = y.reshape(1, 1, 1)
#             theta_list.append(theta)
#         return theta_list, y_list


class MetaQAOA(nn.Cell):
    """
    生成 MetaQAOA网络, 把LSTM看成主体结构, QNN也包含在内
    QNN需要外部输入
    """
    def __init__(self, T, QNN):
        """
        T times of quantum classical interaction 
        """
        super(MetaQAOA, self).__init__()
        # 初始化LSTM三个参数为input_size, hidden_size (int), num_layers (int)
        self.lstm = nn.LSTM(1, 1, 1, has_bias=True, batch_first=True, bidirectional=False)
        self.T = T
        self.qnn = QNN

    def construct(self): # 
        """
        output:
        1. theta_list [T, 2*P]
        2. y_list [T, (1)] 用于计算loss        
        """
        # batch_size, seq_len, input_size
        theta = Tensor(np.ones([1, 2 * P, 1]).astype(np.float32))
        # num_directions * num_layers, batch_size, hidden_size
        h = Tensor(np.zeros([1, 1, 1]).astype(np.float32))
        y = Tensor(np.zeros([1, 1, 1]).astype(np.float32))

        theta_list = []
        y_list = []
        for i in range(self.T):
            theta, (h, _) = self.lstm(theta, (h, y))
            y = self.qnn(theta.reshape(1, -1))
            y_list.append(y.squeeze())
            y = y.reshape(1, 1, 1)
            theta_list.append(theta)
        return theta_list, y_list



class MetaQAOALoss(nn.LossBase):
    def __init__(self):
        super(MetaQAOALoss, self).__init__()
        self.addn = ops.AddN()

    def construct(self, y_list):
        length = len(y_list)
        min_temp = y_list[0]
        # loss_list = [Tensor(np.zeros([], dtype=np.float32))]
        loss_list = [min_temp]
        for i in range(1, length):
            loss_t = y_list[i] - min_temp
            if loss_t < 0:
                loss_list.append(loss_t)
            if y_list[i] < min_temp:
                min_temp = y_list[i]
        loss = self.addn(loss_list)
        return self.get_loss(loss)


if __name__ == "__main__":
    # 测试生成图
    P = 5
    batch_size = 1
    g = gene_random_instance(4)
    print(g)
    print(g.nodes)
    print(g.edges)
    # 测试hc hm and ansatz
    # hc = gene_Hc_circuit(g, 'g0')
    # print(hc)
    # hm = gene_Hm_circuit(g, 'm0')
    # print(hm)
    # ansatz = gene_qaoa_ansatz(g, 3)
    # print(ansatz)
    # 测试forward 过程
    qnn = gene_qaoa_layers(g, P)
    metaqaoa = MetaQAOA(5, qnn)
    # theta = Tensor(np.ones([batch_size, 2 * P, 1]).astype(np.float32))
    # h = Tensor(np.zeros([1, batch_size, 1]).astype(np.float32))
    # y = Tensor(np.ones([1, batch_size, 1]).astype(np.float32))

    # theta_list, y_list = metaqaoa(theta, h, y, qnn)
    theta_list, y_list = metaqaoa()


    print([t.shape for t in theta_list])
    for y in y_list:
        print(y)

    loss_fn = MetaQAOALoss()
    loss = loss_fn(y_list)
    print(loss)