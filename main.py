from src.utils import gene_random_instance
from src.metaqaoa import gene_qaoa_ansatz, gene_random_instance


if __name__ == "__main__":
    # 生成训练数据，n_train in [6, 12], 说是生成500个训练数据， 其中250个测试
    g = gene_random_instance(4)
    print(g)
    print(g.nodes)
    print(g.edges)
    # 生成测试数据，n_test in [8, 14], 100个测试数据, n_train < n_test

    batch_size = 64
    epochs = 1500
    ansatz = gene_qaoa_ansatz(g, 3)
    print(ansatz)
    ansatz.summary()