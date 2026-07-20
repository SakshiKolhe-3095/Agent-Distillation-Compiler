"""
Trains the complexity router on labeled trajectory data, evaluates accuracy
on a held-out split, saves the fitted router.
"""
import json
import random
from inference.router import ComplexityRouter

random.seed(42)

DATA_FILE = "datasets/raw/router_training_data.json"
MODEL_OUT = "models/router.pkl"


def main():
    data = json.load(open(DATA_FILE))
    random.shuffle(data)

    split = int(len(data) * 0.8)
    train_data, test_data = data[:split], data[split:]

    router = ComplexityRouter()
    router.fit([d["problem"] for d in train_data], [d["label"] for d in train_data])

    correct = 0
    for d in test_data:
        pred = router.predict(d["problem"])
        actual = "teacher" if d["label"] == 1 else "student"
        if pred == actual:
            correct += 1

    accuracy = correct / len(test_data)
    print(f"Router accuracy on held-out set: {correct}/{len(test_data)} = {accuracy:.1%}")

    router.save(MODEL_OUT)
    print(f"Saved router to {MODEL_OUT}")


if __name__ == "__main__":
    main()