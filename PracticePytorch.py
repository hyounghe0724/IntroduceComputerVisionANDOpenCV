import torch

w = torch.tensor(1.0, requires_grad=True)
a = w*3
b = a ** 2
a.backward()
print("를 w로 미분한 값은 {}".format(w.grad))