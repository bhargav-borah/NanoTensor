import math
import random

class Value:

  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f'Value(data={self.data})'

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out

  def __radd__(self, other):
    return self + other

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out

  def __rmul__(self, other):
    return self * other

  def __truediv__(self, other):
    return self * (other**-1)

  def __pow__(self, other):
    assert isinstance(other, (int, float)), 'Only supporting int/float powers for now'
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
      self.grad += other * (self.data**(other - 1)) * out.grad
    out._backward = _backward

    return out

  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self,), 'exp')

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward

    return out

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out

  def relu(self):
    x = self.data
    out = Value(0 if x < 0 else x, (self,), 'relu')

    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out

  def sigmoid(self):
    x = self.data
    x = max(min(x, 700), -700)  # Clamping to avoid overflow
    s = 1 / (1 + math.exp(-x))
    out = Value(s, (self,), 'sigmoid')

    def _backward():
      self.grad += s * (1 - s) * out.grad
    out._backward = _backward

    return out

  def elu(self, alpha=1.0):
    x = self.data
    out = Value(x if x > 0 else alpha * (math.exp(x) - 1), (self,), 'elu')

    def _backward():
      self.grad += (1 if x > 0 else alpha * math.exp(x)) * out.grad
    out._backward = _backward

    return out

  def hard_shrink(self, _lambda=0.5):
    x = self.data
    out = Value(x * (x > _lambda) + x * (x < -_lambda), (self,), 'hard_shrink')

    def _backward():
      self.grad += ((x > _lambda) + (x < _lambda)) * out.grad
    out._backward = _backward

    return out

  def hard_sigmoid(self):
    x = self.data
    out = Value((x >= 3) + ((x / 6) + 0.5) * (x > -3 and x < 3), (self,), 'hard_sigmoid')

    def _backward():
      self.grad += ((1 / 6) * (x > -3 and x < 3)) * out.grad
    out._backward = _backward

    return out

  def hard_tanh(self, min_val=-1.0, max_val=1.0):
    x = self.data
    out = Value(max_val * (x > max_val) + min_val * (x < min_val) + x * (x >= min_val and x <= max_val), (self,), 'hard_tanh')

    def _backward():
      self.grad += (x >= min_val and x <= max_val) * out.grad
    out._backward = _backward

    return out

  def hardswish(self):
    x = self.data
    out = Value(x * (x >= 3) + (x * (x + 3) / 6) * (x > -3 and x < 3), (self,), 'hardswish')

    def _backward():
      self.grad += ((x >= 3) + ((2 * x + 3) / 6) * (x > -3 and x < 3)) * out.grad
    out._backward = _backward

    return out

  def leaky_relu(self, negative_slope=0.01):
    x = self.data
    out = Value(x if (x >= 0) else negative_slope * x, (self,), 'leaky_relu')

    def _backward():
      self.grad += (1 if (x >= 0) else negative_slope) * out.grad
    out._backward = _backward

    return out

  def log_sigmoid(self):
    x = self.data
    s = 1 / (1 + math.exp(-x))
    out = Value(math.log(s), (self,), 'log_sigmoid')

    def _backward():
      self.grad += (1 - s) * out.grad
    out._backward = _backward

    return out

  def relu_6(self):
    x = min(max(self.data, 0), 6)
    out = Value(x, (self,), 'relu_6')

    def _backward():
      self.grad += (x > 0 and x < 6) * out.grad
    out._backward = _backward

    return out

  def rrelu(self, lower=0.25, upper=0.3333333333333333, training=True):
    if training:
      self.slope = random.uniform(lower, upper)
    else:
      self.slope = (lower + upper) / 2

    x = self.data
    out = Value(x if x > 0 else self.slope * x, (self,), 'rrelu')

    def _backward():
      self.grad += (1 if x > 0 else self.slope) * out.grad
    out._backward = _backward

    return out

  def selu(self):
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    x = scale * (max(0, self.data) + min(0, alpha * (math.exp(self.data) - 1)))
    out = Value(x, (self,), 'selu')

    def _backward():
      self.grad += (scale * (self.data > 0) + (x <= 0) * (scale * alpha * math.exp(x))) * out.grad
    out._backward = _backward

    return out

  def celu(self, alpha=1.0):
    x = max(0, self.data) + min(0, alpha * (math.exp(self.data / alpha) - 1))
    out = Value(x, (self,), 'celu')

    def _backward():
      self.grad += (1 if self.data > 0 else math.exp(self.data / alpha)) * out.grad
    out._backward = _backward

    return out

  def gelu(self):
    x = self.data
    out = Value(0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))), (self,), 'gelu')

    def _backward():
        sech_squared = 4 / ((math.exp(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)) + math.exp(-math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))) ** 2)
        self.grad += sech_squared * (math.sqrt(2 / math.pi)) * (1 + 3 * 0.044715 * x**2) * out.grad
    out._backward = _backward

    return out

  def silu(self):
    x = self.data
    s = 1 / (1 + math.exp(-x))
    out = Value(x * s, (self,), 'silu')

    def _backward():
      self.grad += s * (1 + x * (1 - s)) * out.grad
    out._backward = _backward

    return out

  def softplus(self, beta=1.0, threshold=20.0):
    x = self.data
    out_val = x if x > threshold else (1 / beta) * math.log(1 + math.exp(beta * x))
    out = Value(out_val, (self,), 'softplus')

    def _backward():
      self.grad += (1 if x > threshold else (1 / (1 + math.exp(-beta * x)))) * out.grad
    out._backward = _backward

    return out

  def mish(self, beta=1.0):
    x = self.data
    softplus_out = (1 / beta) * math.log(1 + math.exp(beta * x))
    out_val = x * math.tanh(softplus_out)
    out = Value(out_val, (self,), 'mish')

    def _backward():
      self.grad += math.tanh(softplus_out) + x * (1 / (1 + math.exp(-x))) * (1 - math.tanh(softplus_out) ** 2) * out.grad
    out._backward = _backward

    return out

  def softshrink(self, _lambda=0.5):
    x = self.data
    out = Value((x - _lambda) * (x > _lambda) + (x + _lambda) * (x < -_lambda), (self,), 'softshrink')

    def _backward():
      self.grad += ((x > _lambda) + (x < -_lambda)) * out.grad
    out._backward = _backward

    return out

  def softsign(self):
    x = self.data
    out = Value(x / (1 + abs(x)), (self,), 'softsign')

    def _backward():
      self.grad += (1 / (1 + abs(x)) ** 2) * out.grad
    out._backward = _backward

    return out

  def tanhshrink(self):
    x = self.data
    out = Value(x - math.tanh(x), (self,), 'tanhshrink')

    def _backward():
      self.grad += (math.tanh(x) ** 2) * out.grad
    out._backward = _backward

    return out

  def backward(self):

    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right

  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot
class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x, activation=None):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        if activation is None or activation == 'linear':
          out = act
        elif activation == 'relu':
          out = act.relu()
        elif activation == 'tanh':
          out = act.tanh()
        elif activation == 'sigmoid':
          out = act.sigmoid()
        elif activation == 'elu':
          out = act.elu()
        elif activation == 'hard_shrink':
          out = act.hard_shrink()
        elif activation == 'hard_sigmoid':
          out = act.hard_sigmoid()
        elif activation == 'hard_tanh':
          out = act.hard_tanh()
        elif activation == 'hardswish':
          out = act.hardswish()
        elif activation == 'leaky_relu':
          out = act.leaky_relu()
        elif activation == 'log_sigmoid':
          out = act.log_sigmoid()
        elif activation == 'relu_6':
          out = act.relu_6()
        elif activation == 'rrelu':
          out = act.rrelu()
        elif activation == 'selu':
          out = act.selu()
        elif activation == 'celu':
          out = act.celu()
        elif activation == 'gelu':
          out = act.gelu()
        elif activation == 'silu':
          out = act.silu()
        elif activation == 'softplus':
          out = act.softplus()
        elif activation == 'mish':
          out = act.mish()
        elif activation == 'softshrink':
          out = act.softshrink()
        elif activation == 'softsign':
          out = act.softsign()
        elif activation == 'tanhshrink':
          out = act.tanhshrink()

        return out

    def parameters(self):
      return self.w + [self.b]

class Layer:

    def __init__(self, nin, nout, activation):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x, activation):
        outs = [n(x, activation=activation) for n in self.neurons]
        return outs

    def parameters(self):
      # return [p for neuron in self.neurons for p in neuron.parameters()]
      params = []
      for neuron in self.neurons:
        params.extend(neuron.parameters())

      return params

class MLP:

    def __init__(self, nin, nouts, activations):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], activation=activations[i]) for i in range(len(nouts))]
        self.activations = activations

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x, activation=self.activations[i])
        return x if len(x) > 1 else x[0]

    def parameters(self):
      return [p for layer in self.layers for p in layer.parameters()]

    def fit(self, xs, ys, n_epochs=1000, verbose=False):
      for epoch in range(n_epochs):
        ypred = [self(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        for p in self.parameters():
          p.grad = 0.0

        loss.backward()

        for p in self.parameters():
          p.data -= 0.01 * p.grad

        if verbose:
          print(f'Epoch {epoch + 1}, Loss: {loss.data}')
