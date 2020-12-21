import math

import torch
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import init

# import torch.nn.functional as F
# from temp import Linear


def compute_inn(x,y):
	p = x * y
	return p.sum(dim=-1) # shape = [batch_size, max_token]

def batch_scalar_mul(s, x):
	bs = s.shape[0]
	for i in range(bs):
		x[i] *= s[i]
	return x


def cplus_2d(x, y, c=0.):
	# batch_size independent version
	# text data: x.shape = [max_token, embed_dim]
	# y = bias, y.shape = embed_dim

	# inn = compute_inn(x_f, y_f)
	# normsq_x = compute_inn(x_f, x_f)
	# normsq_y = compute_inn(y_f, y_f)

	inn = compute_inn(x, y) # shape = max_token
	normsq_x = compute_inn(x, x)
	normsq_y = compute_inn(y, y)

	coef_x = 1 + 2 * c * inn + c * normsq_y # shape = max_token
	coef_y = 1 - c * normsq_x

	# print(x.shape, coef_x.shape, y.shape, coef_y.shape)
	# print(normsq_y)

	# z = batch_scalar_mul(coef_x, x) + batch_scalar_mul(coef_y, y)
	# zx, zy = batch_scalar_mul(coef_x, x), y * coef_y
	# zx, zy = coef_x * x, y * coef_y
	zx = coef_x * x.t() # shape = [embed_dim, max_token]
	# print(zx.shape)
	# zy = coef_y.matmul(y) #
	zy = torch.ger(coef_y,y) # shape = [max_token, embed_dim]
	# print (zx.shape, zy.shape)
	# z = batch_scalar_mul(coef_x, x) + y * coef_y
	# z = y * coef_y
	z = zx.t() + zy # shape = [max_token, embed_dim]

	denom = 1+ c * inn + c * c * normsq_x * normsq_y

	# print(z.shape, denom.shape)

	# return batch_scalar_mul(1/denom, z)
	return (z.t()/denom).t()





def cplus(x, y, c):
	# 3D version
	# torch data format: [sample_index, x1, x2, ...]
	# text data: x.shape = [batch_size, max_token, embed_dim]
	# y = bias, y.shape=embed_dim

#	print(c)

	inn = compute_inn(x, y) # shape = [batch_size, max_token]
	normsq_x = compute_inn(x, x)
	normsq_y = compute_inn(y, y)

	coef_x = 1 + 2 * c * inn + c * normsq_y # shape = [batch_size, max_token]
	coef_y = 1 - c * normsq_x

	# print(x.shape, coef_x.shape, y.shape, coef_y.shape)
	# print(normsq_y)

	# z = batch_scalar_mul(coef_x, x) + batch_scalar_mul(coef_y, y)
	# zx, zy = batch_scalar_mul(coef_x, x), y * coef_y
	def tr(x):
		return x.transpose(dim0=1, dim1=2).transpose(dim0=0, dim1=1)

	# zx = coef_x * tr(x) # shape = [embed_size, batch_size, max_token]
	zx = torch.einsum('ijk,ij->ijk', (x, coef_x))
	# zy = torch.ger(coef_y, y) # shape = [batch_size, max_token, embed_dim]
	zy = torch.einsum('ij,k->ijk', (coef_y, y))
	# print (zx.shape, zy.shape)
	# zy = tr(torch.ger(coef_y, y)) # shape = [embed_dim, batch_size, max_token]
	# z = batch_scalar_mul(coef_x, x) + y * coef_y
	# z = y * coef_y
	z = zx + zy # shape = [batch_size, max_token, embed_dim]

	denom = 1+ c * inn + c * c * normsq_x * normsq_y # shape = [batch_size, max_token]

	# print(z.shape, denom.shape)

	# return batch_scalar_mul(1/denom, z)
	# return (z/denom).transpose(dim0=0,dim1=2) # shape = [batch_size, max_token, embed_dim]
	return torch.einsum('ijk,ij->ijk',(z,1/denom))





class cLinear(Module):

    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, c, bias=True):
        super(cLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.c = c
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # return F.linear(input, self.weight, self.bias)
        # w = input.matmul(self.weight.t())
        # print(input.shape, self.weight.shape, w.shape, self.bias.shape)
        return cplus(input.matmul(self.weight.t()), self.bias, self.c)
        # return cplus(input.matmul(self.weight.t()), self.bias, self.c) - F.linear(input, self.weight, self.bias) # for testing

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



def error(x, y, c=0.):

	z = cplus(x,y,c)
	s = x + y

	e = torch.abs(x + y - z).sum()
	rel_e = e/s.abs().sum()

	return rel_e


def test_cplus():
	# x = torch.Tensor([[1,2,3,3],[2,3,5,6]]).t()

	x = torch.Tensor([
					[[1,2],[3,3]], 
					[[2,2],[3,3]],
					[[2,2],[3,3]]])

	y = torch.Tensor([3,5])


	c = 0.000

	# print(x.shape, y.shape)
	print(x + y, "\n", cplus(x,y,c))

	print(error(x,y,c))



def test_cLin():
	# x = torch.Tensor([[1,2,3,3],[2,3,5,6]])
	x = torch.Tensor([
					[[1,2],[3,3]], 
					[[2,2],[3,3]],
					[[2,2],[3,3]]])
	c = 0.000

	# lin = Linear(in_features=x.shape[1], out_features=5)
	clin = cLinear(in_features=x.shape[1], out_features=5, c=c)

	y = clin(x)
	# y = lin(x)
	# print(x.shape)
	# print(y-cy)
	# print(lin.weight.data,clin.weight.data)
	print(y)





if __name__ =='__main__':

	# test_cplus()
	# torch.manual_seed(0)

	test_cLin()
	
