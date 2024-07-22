"""
Exercises & Extra-curriculum

See exercises for this notebook here: https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises
"""
import torch

"""
1. Documentation reading - A big part of deep learning (and learning to code in general) is getting familiar with the 
documentation of a certain framework you're using. We'll be using the PyTorch documentation a lot throughout the rest 
of this course. So I'd recommend spending 10-minutes reading the following (it's okay if you don't get some things for 
now, the focus is not yet full understanding, it's awareness). See the documentation on 
torch.Tensor - https://pytorch.org/docs/stable/tensors.html#torch-tensor
and for 
torch.cuda. - https://pytorch.org/docs/master/notes/cuda.html#cuda-semantics
"""

"""
2. Create a random tensor with shape (7, 7).
"""
x = torch.rand(7, 7)
print(x, x.shape) # torch.Size([7, 7])

"""
3. Perform a matrix multiplication on the tensor from 2. with another random tensor with shape (1, 7) 
(hint: you may have to transpose the second tensor).
"""
y = torch.rand(1, 7)
# print(torch.mul(y, x)) # inner dimensions are the same
# Or if we want to transpose for x and y
z = torch.matmul(x, y.T)
print(z, z.shape)

"""
4. Set the random seed to 0 and do exercises 2 & 3 over again.
"""
torch.manual_seed(0)
x = torch.rand(7, 7)
y = torch.rand(1, 7)
# Or if we want to transpose
z = torch.matmul(x, y.T)
print(z, z.shape)

"""
5. Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? 
(hint: you'll need to look into the documentation for torch.cuda for this one). 
If there is, set the GPU random seed to 1234.
"""
torch.cuda.manual_seed(1234)

"""
6. Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this). 
Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed).
"""
torch.manual_seed(1234)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
x = torch.rand(2, 3).to(device)
y = torch.rand(2, 3).to(device)
print(x, y)

"""
7. Perform a matrix multiplication on the tensors you created in 6 
(again, you may have to adjust the shapes of one of the tensors).
"""
z = torch.matmul(x, y.T)
print(z)

"""
8. Find the maximum and minimum values of the output of 7.
"""
print(z.max())
print(z.min())

"""
9. Find the maximum and minimum index values of the output of 7.
"""
print(z.argmax())
print(z.argmin())

"""
10. Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed to be 
left with a tensor of shape (10). Set the seed to 7 when you create it and print out the first tensor and it's shape 
as well as the second tensor and it's shape.
"""
torch.manual_seed(7)
x = torch.rand(1, 1, 1, 10)
print(x, x.shape)
y = x.squeeze()
print(y, y.shape)
