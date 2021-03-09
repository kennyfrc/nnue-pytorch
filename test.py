import torch

base = torch.randn(5,2)

mask = base.gt(0.5)

print(base)
print(mask)

print(torch.where(base.gt(0.5), base*100, base))

lambda_ = torch.tensor([0.5, 1.0, 0.0, 0.1])
q_score = torch.tensor([0.5, 0.1, 0.2, 0.5])
z_score = torch.tensor([1.0, 0.5, 0.5, 1.0])
outcome = torch.tensor([1.0, 0.5, 0.5, 1.0])


# teacher score
t_score = torch.where(
  torch.logical_and(lambda_.lt(1), outcome.eq(1)),
  (q_score * lambda_) + (z_score * (1 - lambda_)),
  q_score
)

print(t_score)