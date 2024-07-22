"""
01. Pytorch Fundamentals

Resource notebook: https://www.learnpytorch.io/00_pytorch_fundamentals/

If you have a question: https://github.com/mrdbourke/pytorch-deep-learning/discussions
"""
import torch
import numpy as np
print(torch.__version__)

"""
Introduction to Tensors

Creating tensors
Pytorch tensors are created using torch.tensor() = https://pytorch.org/docs/stable/tensors.html
"""
# Scalar
scalar = torch.tensor(7)
print(scalar) # tensor(7)
# It doesn't have dimensions
print(scalar.ndim) # 0
# Get tensor back as Python int
print(scalar.item()) # 7

# Vector
vector = torch.tensor([7, 7])
print(vector) # tensor([7, 7])
print(vector.ndim) # 1
print(vector.shape) # torch.Size([2])

# MATRIX
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
print(MATRIX)
print(MATRIX.ndim) # 2
print(MATRIX.shape) # torch.Size([2, 2])
print(MATRIX[0]) # tensor([7, 8])

# TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(TENSOR)
print(TENSOR.ndim) # 3
print(TENSOR.shape) #  torch.Size([1, 3, 3])
print(TENSOR[0])

"""
Random tensors

Why random tensors?
Random tensors are important because the way many neural networks learn
is that they start with tensors full of random numbers and then adjust
those random numbers to better represent the data
Starts with random numbers -> look at data -> update random numbers
-> look at data -> update random numbers -> .....
Torch random tensors: https://pytorch.org/docs/stable/generated/torch.rand.html
"""
# Create a random tensor of size (3, 4)
random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.ndim) # 2
# Create a random tensor with similar shape to an image tensor
# size is the first parameter
random_image_size_tensor = torch.rand(size=(3, 224, 224)) # color channel, height, widht
print(random_image_size_tensor.shape) # torch.Size([3, 224, 224])
print(random_image_size_tensor.ndim) # 3

"""
Zeros and ones
"""
# Create a tensor of all zeros
zeros = torch.zeros(3, 4)
print(zeros)
# Create a tensor of all ones
ones = torch.ones(3, 4)
print(ones)
print(ones.dtype) # torch.float32
print(random_tensor.dtype) # torch.float32

"""
Create a range of tensors and tensors-like
"""
# Use torch.arange(), range() is deprecated
one_to_ten = torch.arange(start=1, end=11, step=1)
print(one_to_ten) # tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
# Creating tensors like
ten_zeros = torch.zeros_like(one_to_ten)
print(ten_zeros) # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

"""
Tensor datatypes

Tensor datatypes is one of the 3 big errors you'll run into with PyTorch
& deep learning:
1. Tensors not right datatype
2. Tensors not right shape
3. Tensors not on the right device

Precision in computing: https://en.wikipedia.org/wiki/Precision_(computer_science)
"""
# Float 32 tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # what data type is the tensor
                               device=None, # what device is your tensor on
                               requires_grad=False) # whether to track gradients with these tensors operations
print(float_32_tensor) # tensor([3., 6., 9.])
print(float_32_tensor.dtype) # torch.float32
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor) # tensor([3., 6., 9.], dtype=torch.float16)
print(float_16_tensor * float_32_tensor) # tensor([ 9., 36., 81.])

int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
print(int_32_tensor) # tensor([3, 6, 9], dtype=torch.int32)
print(float_16_tensor * int_32_tensor) # tensor([ 9., 36., 81.], dtype=torch.float16)

"""
Getting information from tensors (tensor attributes)

1. Tensors not right datatype - to do get datatype from a tensor, can use 'tenros.dtype'
2. Tensors not right shape - to get shape from a tensor, can use 'tensor.shape'
3. Tensors not on the right device - to get device from a tensor, can use 'tensor.device'
"""
# Create a tensor
some_tensor = torch.rand(3, 4)
print(some_tensor)
# Find out details about some tensor
print(f'Datatype of tensor: {some_tensor.dtype}') # torch.float32
print(f'Shape of tensor: {some_tensor.shape}') # torch.Size([3, 4])
print(f'Device tensor is on: {some_tensor.device}') # cpu

"""
Manipulation tensors (tensor operations)

Tensor operations include:
* Addition
* Substraction
* Multiplication (element-wise)
* Division
"""
# Create a tensor and add 10 to it
tensor = torch.tensor([1, 2, 3])
print(tensor + 10) # tensor([11, 12, 13])
# Multiply tensor by 10
print(tensor * 10) # tensor([10, 20, 30])
# Substract 10
print(tensor - 10) # tensor([-9, -8, -7])
# Division 10
print(tensor / 10) # tensor([0.1000, 0.2000, 0.3000])
# Try out PyTorch in-built functions
print(torch.add(tensor, 10)) # tensor([11, 12, 13])
print(torch.mul(tensor, 10)) # tensor([10, 20, 30])

"""
Matrix multiplication

Two main ways to performing multiplication in neural networks and deep learning:
1. Element-wise multiplication
2. Matrix multiplication (dot product)

More information on multiplyng matrices - https://www.mathsisfun.com/algebra/matrix-multiplying.html

There are two main rules that performing matrix multiplication needs to satisfy:
1. The INNER DIMENSIONS must match:
* (3, |2|) @ (|3|, 2) won't work (2, 3) - @ is matrix multiplication
* (2, |3|) @ (|3|, 2) will work (3, 3)
* (3, |2|) @ (|2|, 3) will work (2, 2)
2. The resulting matrix has the shape of the OUTER DIMENSIONS:
* (|2|, 3) @ (3, |2|) -> (2, 2)
* (|3|, 2) @ (2, |3|) -> (3, 3)
"""
# Element wise multiplication
print(tensor, "*", tensor) # tensor([1, 2, 3]) * tensor([1, 2, 3])
print(f'Equals: {tensor * tensor}') # Equals: tensor([1, 4, 9])
# Matrix multiplication (faster) = 246 μs - micro
print(torch.matmul(tensor, tensor)) # tensor(14)
# Matrix multiplication by hand
print((1*1) + (2*2) + (3*3)) # 14
# Matrix multiplication for loop = 2.01 ms - milli
value = 0
for i in range(len(tensor)):
    value += tensor[i] * tensor[i]
print(value) # tensor(14)

"""
One of the most common errors in deep learning: shape errors
"""
# Shapes for matrix multiplication
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])
tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])
# torch.mm is an alias of torch.matmul
# print(torch.mm(tensor_A, tensor_B)) # shape error

# To fix our tensor shape issue, we can manipulate the shape of one
# of our tensors using a "transpose"
# A transpose switches the axes or dimensions of a given tensor
print(tensor_B.T) # shape = 2, 3
print(tensor_B) # shape = 3, 2
# The matrix multiplication operation works when tensor_B is transposed
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}")
print(f"New shapes: tensor_A = {tensor_A.shape} (same shape as above), tensor_B = {tensor_B.T.shape}")
print(f"Multiplying: {tensor_A.shape} @ {tensor_B.T.shape} <- inner dimensions must match")
output = torch.matmul(tensor_A, tensor_B.T)
print(f'Output: \n{output}')
print(f"Output shape: {output.shape}") # torch.Size([3, 3])

"""
Finding the min, max, mean, sum, etc (tensor aggregation)
"""
# Create a tensor
x = torch.arange(0, 100, 10)
print(x) # tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
# Find min
print(torch.min(x), x.min()) # tensor(0)
# Find max
print(torch.max(x), x.max()) # tensor(90)
# Find mean - x is not the right datatype (int64) for now
# Input dtype must be either a floating point or complex dtype. Got: Long
print(torch.mean(x.type(torch.float32)), x.type(torch.float32).mean()) # tensor(45.)
# Find sum
print(torch.sum(x), x.sum()) # tensor(450)

"""
Finding the positional min and max
"""
# Find the position in tensor that has the minimum value with argmin() -> returns index position of target tensor where the minimum value occurs
x = torch.arange(1, 100, 10)
print(x) # tensor([ 1, 11, 21, 31, 41, 51, 61, 71, 81, 91])
print(x.argmin()) # tensor(0) -> index
# Find the position in tensor that has the maximum value with argmax()
print(x.argmax()) # tensor(9)

"""
Reshaping, stacking, squeezing and unsqueezing tensors
* Reshaping - Reshapes an input tensor to a defined shape
* View - Return a view of an input tensor of certain shape but keep the same memory as the original tensor
* Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
* Squeeze - removes all '1' dimensions from a tensor
* Unsqueeze - add '1' dimension to a target tensor
* Permute - Return a view of input with dimension permuted (swapped) in a certain way
"""
# Let's create a tensor
x = torch.arange(1., 10.)
print(x, x.shape) # tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.]) torch.Size([9])
# Add an extra dimension
# x_reshaped = x.reshape(1, 7) - shape '[1, 7]' is invalid for input of size 9
x_reshaped = x.reshape(1, 9)
print(x_reshaped, x_reshaped.shape) # tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9.]]) torch.Size([1, 9])
# x_reshaped = x.reshape(2, 9) - shape '[2, 9]' is invalid for input of size 9
x_reshaped = x.reshape(9, 1)
print(x_reshaped, x_reshaped.shape) # ... torch.Size([9, 1])
# Change the view
z = x.view(1, 9)
print(x, z.shape) # tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.]) torch.Size([1, 9])
# Changing z changes x (because a view of a tensor shares te same memory as the original)
z[:, 0] = 5
print(z, x) # tensor([[5., 2., 3., 4., 5., 6., 7., 8., 9.]]) -> x2
# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked) # tensor([[5., 2., 3., 4., 5., 6., 7., 8., 9.], -> x4 vertically

# torch.squeeze() - removes all single dimensions from a target tensor
print(x_reshaped)
print(f"Previous shape: {x_reshaped.shape}") # torch.Size([9, 1])
# Remove extra dimensions from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(x_squeezed)
print(f"Squeezed shape: {x_squeezed.shape}") # torch.Size([9])

# torch.unsqueeze() - adds a single dimension to a target tensor at a specific dim (dimension)
print(x_squeezed)
print(f"Previous shape: {x_squeezed.shape}") # torch.Size([9])
# Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0) # Selezioni in quale indice inserire la nuova dimensione
print(x_unsqueezed)
print(f"Unsqueezed shape: {x_unsqueezed.shape}") # torch.Size([1, 9])

# torch.permute - rearranges the dimensions of a target tensor in a specified order
x_original = torch.rand(size=(224, 224, 3)) # h x w x color channel
# Permute the original tensor to rearraange the axis (or dim) order
x_permuted = x_original.permute(2, 0, 1) # shifts the number axis in the new position
print(x_permuted.shape) # torch.Size([3, 224, 224])

"""
Indexing (selecting data from tensors)
Indexing with pytorch is similar to indexing with numpy
"""
# Create a tensor
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x, x.shape)
# Let's index on our new tensor
print(x[0])
# Let's index on the middle bracket (dim 1)
print(x[0][0]) # or [0, 0] - tensor([1, 2, 3])
# Let's index on the most inner bracket (last dim)
print(x[0, 0, 0]) # tensor(1)
# print(x[1, 1, 1]) - index 1 is out of bounds for dimension 0 with size 1
# ricorda la shape - (1, 3, 3)
print(x[0, 2, 2]) # tensor(9)
# You can also use ':' to select "all" of a target dimension
print(x[:, 0]) # tensor([[1, 2, 3]])
# Get all values of 0th and 1st dimensions but only index 1 of 2nd dimension
print(x[:, :, 1]) # tensor([[2, 5, 8]])
# Get all values of 0th dimension only the index 1 values of 1st and 2nd dimension
print(x[:, 1, 1]) # tensor([5]) - same as [0, 1, 1]
# Get index 0 of 0th and 1st dimension and all values of 2nd
print(x[0, 0, :]) # tensor([1, 2, 3]) - same as [0, 0]
# Index on x to return 3, 6, 9
print(x[:, :, 2]) # tensor([[3, 6, 9]])

"""
PyTorch tensors & NumPy

NumPy is a popular scientific Python numerical computing library
And because of this PyTorch has functionality to intercat with it
* Data in NumPy, want in PyTorch tensor -> torch.from_numpy(ndarray)
* PyTorch tensor -> NumPy -> torch.Tensor.numpy()
"""
# numpy array to tensor
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array) # warning: when converting from numpy, pytorch reflects numpy defaul dtype of float 64 unless specified
print(array, tensor) # [1. 2. 3. 4. 5. 6. 7.] tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64)
print(array.dtype, tensor.dtype) # float64 torch.float64
# Change the value of array, what will do to the tensor?
array = array + 1
print(array, tensor) # [2. 3. 4. 5. 6. 7. 8.] tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64)
# Tensor to numpy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor, numpy_tensor) # tensor([1., 1., 1., 1., 1., 1., 1.]) [1. 1. 1. 1. 1. 1. 1.]
print(tensor.dtype, numpy_tensor.dtype) # torch.float32 float32
# Change the tensor, what happens to numpy_tensor?
tensor = tensor + 1
print(tensor, numpy_tensor) # tensor([2., 2., 2., 2., 2., 2., 2.]) [1. 1. 1. 1. 1. 1. 1.]

"""
Reproducibility (trying to take random out of random)
In short how a neural network learns:
Starts with random numbers -> tensor operations -> 
-> update random numbers to try and make them better representation of the data ->
-> again -> again -> .....
To reduce the randomness in neural networks and pytorch comes the concept of "manual seed"
Extra resources for reproducibility:
* https://pytorch.org/docs/stable/notes/randomness.html
* https://en.wikipedia.org/wiki/Random_seed
"""
# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)
print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B) # all different values likely
# Let's make some random but reproducible tensors
# Set the random seed
MANUAL_SEED = 77
torch.manual_seed(MANUAL_SEED)
random_tensor_C = torch.rand(3, 4)
random_tensor_D = torch.rand(3, 4)
print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D) # still different values
# But let's call the seed each time
torch.manual_seed(MANUAL_SEED)
random_tensor_C = torch.rand(3, 4)
torch.manual_seed(MANUAL_SEED)
random_tensor_D = torch.rand(3, 4)
print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D) # same values

"""
Running tensors and pytorch on the GPUs (and making faster computations)
GPU = faster computation on number thanks to CUDA + NVIDIA hardware + pytorch working behind the scenes
to make everything hunky dory (good)

GO SEE THE COLAB GOOGLE FILE 00
"""
