import math
import random

import minitorch
from project.run_scalar import ScalarTrain


def run():
    for name in ["Simple", "Diag", "Split", "Xor"]:
        random.seed(0)
        data = minitorch.datasets[name](50)
        trainer = ScalarTrain(10)
        losses = []

        def log(epoch, total_loss, correct, history):
            assert all(math.isfinite(value) for value in history)
            losses[:] = history
            print(f"epoch={epoch:03d} loss={total_loss:.6f} correct={correct}/{data.N}")

        print(f"dataset={name} seed=0 points=50 hidden=10 rate=0.5 epochs=500")
        trainer.train(data, 0.5, max_epochs=500, log_fn=log)
        predictions = [trainer.run_one(x).data for x in data.X]
        correct = sum((p > 0.5) == y for p, y in zip(predictions, data.y))
        final_loss = sum(
            -minitorch.operators.log(p if y else 1.0 - p)
            for p, y in zip(predictions, data.y)
        )
        assert all(math.isfinite(p) for p in predictions)
        assert math.isfinite(final_loss) and final_loss < losses[0]
        print(f"initial_loss={losses[0]:.6f} final_loss={final_loss:.6f} final_correct={correct}/{data.N}")
        print()


if __name__ == "__main__":
    run()
