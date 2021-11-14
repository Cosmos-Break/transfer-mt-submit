import matplotlib.pyplot as plt
import numpy as np
# emb_mt_name = 'opus-mt-de-en/embedding'
# vocab_size, emb_size = 50000, 512
# emb_list = []
# for line in open(emb_mt_name):
#     line = line.strip().split()
#     if len(line) == 2:
#         continue
#     token = line[0]
#     emb = line[1:]
#     emb = [float(x) for x in emb]
#     assert len(emb) == 512
#     emb_list.append(emb)
# emb_list = np.array(emb_list)
# cov = np.cov(emb_list.T)

# mean = np.mean(emb_list)
# var = np.var(emb_list)
# print(mean, var)
# # 0.0013373970354963827 0.0022201915919596235
# mean = np.mean(emb_list, axis=0)
# # a = np.random.multivariate_normal(mean, cov, size=1).tolist()[0]
# cov_arr = np.random.multivariate_normal(mean, cov, size=vocab_size)
# 0.0013373970354963827 0.0022201915919596235
mean = [0, 0]
cov = [[1, 0], [0, 100]]  # diagonal covariance
# x, y = np.random.multivariate_normal(mean, cov, 50000).T
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

x, y = np.random.normal(0.0013373970354963827, 0.0022201915919596235, 2)
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()
plt.savefig('fig.png')