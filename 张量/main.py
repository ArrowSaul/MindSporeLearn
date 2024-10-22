import numpy as np
import mindspore
from mindspore import ops
from mindspore import Tensor, CSRTensor, COOTensor

# 创建张量
data = [1, 0, 1, 0]
x_data = Tensor(data)
print(x_data, x_data.shape, x_data.dtype)
np_array = np.array(data)
x_np = Tensor(np_array)
print(x_np, x_np.shape, x_np.dtype)
# 使用init初始化器构造张量
from mindspore.common.initializer import One, Normal

# Initialize a tensor with ones
tensor1 = mindspore.Tensor(shape=(2, 2), dtype=mindspore.float32, init=One())
# Initialize a tensor from normal distribution
tensor2 = mindspore.Tensor(shape=(2, 2), dtype=mindspore.float32, init=Normal())

print("tensor1:\n", tensor1)
print("tensor2:\n", tensor2)
# 继承另一个张量的属性，形成新的张量
from mindspore import ops

x_ones = ops.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_zeros = ops.zeros_like(x_data)
print(f"Zeros Tensor: \n {x_zeros} \n")
# 张量的属性
x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.int32)

print("x_shape:", x.shape)
print("x_dtype:", x.dtype)
print("x_itemsize:", x.itemsize)
print("x_nbytes:", x.nbytes)
print("x_ndim:", x.ndim)
print("x_size:", x.size)
print("x_strides:", x.strides)
# 张量索引
tensor = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
print("First row: {}".format(tensor[0]))
print("value of bottom right corner: {}".format(tensor[1, 1]))
print("Last column: {}".format(tensor[:, -1]))
print("First column: {}".format(tensor[..., 0]))
# 张量计算
x = Tensor(np.array([1, 2, 3]), mindspore.float32)
y = Tensor(np.array([4, 5, 6]), mindspore.float32)

output_add = x + y
output_sub = x - y
output_mul = x * y
output_div = y / x
output_mod = y % x
output_floordiv = y // x

print("add:", output_add)
print("sub:", output_sub)
print("mul:", output_mul)
print("div:", output_div)
print("mod:", output_mod)
print("floordiv:", output_floordiv)
# 张量拼接
# concat将给定维度上的一系列张量连接起来。
data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.concat((data1, data2), axis=0)

print(output)
print("shape:\n", output.shape)
# stack则是从另一个维度上将两个张量合并起来。
data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.stack([data1, data2])
print(output)
print("shape:\n", output.shape)
# Tensor转换为NumPy
t = Tensor([1., 1., 1., 1., 1.])
print(f"t: {t}", type(t))
n = t.asnumpy()
print(f"n: {n}", type(n))
# NumPy转换为Tensor
n = np.ones(5)
t = Tensor.from_numpy(n)
np.add(n, 1, out=n)
print(f"n: {n}", type(n))
print(f"t: {t}", type(t))
# CSRTensor
indptr = Tensor([0, 1, 2])
indices = Tensor([0, 1])
values = Tensor([1, 2], dtype=mindspore.float32)
shape = (2, 4)

# Make a CSRTensor
csr_tensor = CSRTensor(indptr, indices, values, shape)

print(csr_tensor.astype(mindspore.float64).dtype)
# COOTensor
indices = Tensor([[0, 1], [1, 2]], dtype=mindspore.int32)
values = Tensor([1, 2], dtype=mindspore.float32)
shape = (3, 4)

# Make a COOTensor
coo_tensor = COOTensor(indices, values, shape)

print(coo_tensor.values)
print(coo_tensor.indices)
print(coo_tensor.shape)
print(coo_tensor.astype(mindspore.float64).dtype)  # COOTensor to float64
