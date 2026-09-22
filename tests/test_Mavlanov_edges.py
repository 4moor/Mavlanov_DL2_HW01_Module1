from typing import Callable

import pytest

from minitorch import Module, Parameter, operators


@pytest.mark.task0_1
@pytest.mark.parametrize("x", [-1000.0, -100.0, 0.0, 100.0, 1000.0])
def test_sigmoid_extreme(x: float) -> None:
    assert 0.0 <= operators.sigmoid(x) <= 1.0
    assert operators.sigmoid(x) + operators.sigmoid(-x) == pytest.approx(1.0)


@pytest.mark.task0_1
def test_zero_and_tolerance() -> None:
    assert operators.relu(0.0) == 0.0
    assert operators.relu_back(0.0, 3.0) == 0.0
    assert operators.is_close(0.0, 0.009)
    assert not operators.is_close(0.0, 0.01)


@pytest.mark.task0_1
@pytest.mark.parametrize(
    "fn,back,x",
    [
        (operators.log, operators.log_back, 0.2),
        (operators.log, operators.log_back, 4.0),
        (operators.inv, operators.inv_back, -2.0),
        (operators.inv, operators.inv_back, 0.3),
        (operators.relu, operators.relu_back, -2.0),
        (operators.relu, operators.relu_back, 2.0),
    ],
)
def test_backward_numerically(
    fn: Callable[[float], float],
    back: Callable[[float, float], float],
    x: float,
) -> None:
    eps = 1e-6
    d = -2.5
    expected = (fn(x + eps) - fn(x - eps)) / (2 * eps) * d
    assert back(x, d) == pytest.approx(expected, rel=1e-5, abs=1e-7)


@pytest.mark.task0_3
def test_empty_lists_and_iterators() -> None:
    assert operators.negList([]) == []
    assert operators.addLists([], []) == []
    assert operators.sum([]) == 0.0
    assert operators.prod([]) == 1.0
    assert operators.negList(iter([1.0, -2.0])) == [-1.0, 2.0]
    assert operators.addLists(iter([1.0, 2.0]), iter([3.0, 4.0])) == [4.0, 6.0]
    assert operators.prod(iter([2.0, 3.0, 4.0])) == 24.0
    total = operators.reduce(operators.add, 5.0)
    assert total([1.0, 2.0]) == 8.0
    assert total([]) == 5.0
    assert total([1.0]) == 6.0


@pytest.mark.task0_4
def test_deep_modules() -> None:
    root = Module()
    assert root.parameters() == []
    child = Module()
    leaf = Module()
    root.child = child
    child.leaf = leaf
    parameter = Parameter(7.0)
    leaf.weight = parameter

    assert root.named_parameters() == [("child.leaf.weight", parameter)]
    assert root.parameters()[0] is parameter
    root.eval()
    assert not any(module.training for module in [root, child, leaf])
    root.train()
    assert all(module.training for module in [root, child, leaf])

    parameter.update(9.0)
    assert root.parameters()[0].value == 9.0
