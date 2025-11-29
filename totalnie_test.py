import torch
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh

from torch.distributed.fsdp import MixedPrecisionPolicy

from torch.distributed.fsdp import fully_shard

# Suppose we’re using tensor parallel across 4 GPUs
device_mesh = init_device_mesh("cuda", (2,))

# Example shapes

A = nn.Parameter(torch.randn(1024, 512, device="cuda"))
B = nn.Parameter(torch.randn(512, 256, device="cuda"))

class MyModelA(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters A and B
        self.A = nn.Parameter(torch.randn(1024, 512, device="cuda"))

    def forward(self, x):
        x = x @ self.A  # shape: (*, 512)
        return x

class MyModelB(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters A and B
        self.B = nn.Parameter(torch.randn(512, 256, device="cuda"))

    def forward(self, x):
        x = x @ self.B  # shape: (*, 512)
        return x



class MyModelC(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters A and B
        self.C = nn.Parameter(torch.randn(1024, 256, device="cuda"))

    def forward(self, x):
        x = x @ self.C  # shape: (*, 256)
        return x


# fsdp2_kwargs = {
#     "mp_policy": MixedPrecisionPolicy(
#         param_dtype=torch.bfloat16,
#         reduce_dtype=torch.float32,
#     )
# }

fsdp2_kwargs = {
    "mp_policy": MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )
}


m1 = MyModelA().cuda()
m2 = MyModelB().cuda()
m3 = MyModelC().cuda()
fully_shard(m1, mesh=device_mesh, **fsdp2_kwargs)
fully_shard(m2, mesh=device_mesh, **fsdp2_kwargs)
fully_shard(m3, mesh=device_mesh, **fsdp2_kwargs)


with torch.no_grad():
    C = m1.A @ m2.B  # This works with DTensors as long as shard layouts are compatible

print(C)
# C = C.redistribute(mesh, [Shard(0)])  # or Shard(1), depending on your intended layout


#Copy weights from C to m3.C
with torch.no_grad():
    m3.C.copy_(C)


x = torch.randn(4, 1024, device="cuda")  # Example input
out1 = m3(x)


loss = out1.pow(2).mean()  # Example: simple MSE-like loss
loss.backward()

# Step 4: Recompute C = m1.A @ m2.B, but now with gradient tracking
# We want to propagate the gradient that accumulated in m3.C
# with torch.enable_grad():
C_recomputed = m1.A @ m2.B

# Step 5: Backward through recomputed C, using gradient from m3.C
# (This sends gradient back into A and B)
C_recomputed.backward(m3.C.grad)


# C_local = C.full_tensor()

# out2 =  x @ C_local


print("XD")


print("Grad A:", m1.A.grad.norm().item())
print("Grad B:", m2.B.grad.norm().item())
print("Grad C:", m3.C.grad.norm().item())

print("Done ✅")