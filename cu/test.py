import torch
import cpp_interpolation

feats = torch.randn(2)
point = torch.zeros(2)

out = cpp_interpolation.trilinear_interpolation(feats, point)

print(out)
