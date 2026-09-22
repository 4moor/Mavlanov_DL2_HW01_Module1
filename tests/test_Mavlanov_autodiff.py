import pytest

from minitorch import Parameter, Scalar, SGD, central_difference, topological_sort
from project.run_scalar import Linear, Network


@pytest.mark.task1_1
def test_difference_argument() -> None:
    args = [2.0, 3.0]
    result = central_difference(lambda x, y: x * y * y, *args, arg=1)
    assert result == pytest.approx(12.0)
    assert args == [2.0, 3.0]


@pytest.mark.task1_4
def test_shared_graph() -> None:
    x = Scalar(2.0)
    square = x * x
    out = square * square + 3.0 * square
    out.backward()
    assert out.data == 28.0
    assert x.derivative == pytest.approx(44.0)

    nodes = list(topological_sort(out))
    positions = {node.unique_id: i for i, node in enumerate(nodes)}
    assert len(nodes) == len(positions)
    for node in nodes:
        for parent in node.parents:
            if not parent.is_constant():
                assert positions[node.unique_id] < positions[parent.unique_id]


@pytest.mark.task1_4
def test_explicit_constant() -> None:
    x = Scalar(2.0)
    constant = Scalar(3.0, back=None)
    out = x * constant
    pairs = list(out.chain_rule(1.0))
    assert len(pairs) == 1
    assert pairs[0][0] is x
    assert pairs[0][1] == 3.0
    assert all(node.unique_id != constant.unique_id for node in topological_sort(out))
    out.backward()
    assert x.derivative == 3.0
    assert constant.derivative is None


@pytest.mark.task1_4
def test_deep_graph() -> None:
    x = Scalar(1.0)
    out = x
    for i in range(2000):
        out = out + 1.0
    out.backward()
    assert out.data == 2001.0
    assert x.derivative == 1.0


@pytest.mark.task1_4
def test_accumulation_and_zero_grad() -> None:
    parameter = Parameter(Scalar(3.0))
    optimizer = SGD([parameter], lr=0.1)
    out = parameter.value * parameter.value
    out.backward(2.0)
    out.backward()
    assert parameter.value.derivative == 18.0
    optimizer.step()
    assert parameter.value.data == pytest.approx(1.2)
    (parameter.value * 2.0).backward()
    optimizer.zero_grad()
    assert parameter.value.derivative is None


@pytest.mark.task1_4
def test_reverse_operations() -> None:
    x = Scalar(2.0)
    out = 3.0 - x + 8.0 / x
    out.backward()
    assert out.data == 5.0
    assert x.derivative == -3.0


@pytest.mark.task1_4
def test_comparison_gradients() -> None:
    x, y = Scalar(1.0), Scalar(2.0)
    ((x < y) + (x == y) + (x > y)).backward()
    assert x.derivative == 0.0
    assert y.derivative == 0.0


def test_linear_forward_and_gradients() -> None:
    layer = Linear(2, 1)
    layer.weights[0][0].update(Scalar(2.0))
    layer.weights[1][0].update(Scalar(-1.0))
    layer.bias[0].update(Scalar(0.5))
    x, y = Scalar(3.0), Scalar(4.0)
    out = layer.forward([x, y])[0]
    out.backward()
    assert out.data == 2.5
    assert x.derivative == 2.0
    assert y.derivative == -1.0
    assert layer.weights[0][0].value.derivative == 3.0
    assert layer.weights[1][0].value.derivative == 4.0
    assert layer.bias[0].value.derivative == 1.0


def test_network_parameters() -> None:
    network = Network(3)
    assert len(network.parameters()) == 9 + 12 + 4
    value = network.forward([Scalar(0.2), Scalar(0.8)])
    assert 0.0 <= value.data <= 1.0


def test_optimizer_reduces_loss() -> None:
    parameter = Parameter(Scalar(0.0))
    optimizer = SGD([parameter], lr=0.1)
    for i in range(20):
        optimizer.zero_grad()
        error = parameter.value - 3.0
        loss = error * error
        loss.backward()
        optimizer.step()
    assert abs(parameter.value.data - 3.0) < 0.04
