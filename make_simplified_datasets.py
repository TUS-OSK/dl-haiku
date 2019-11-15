#!/usr/bin/env python3
import argparse
import csv
import json
import random
from typing import List

import numpy as np
from tqdm import tqdm


def addition(a: int, b: int) -> int:
    return a + b


def subtraction(a: int, b: int) -> int:
    return a - b


def multiplication(a: int, b: int) -> int:
    return a * b


def division(a: int, b: int) -> int:
    return a // b


operations = [
    ("+", addition),
    ("-", subtraction),
    ("*", multiplication),
    ("/", division),
]


def make_data() -> str:
    a = np.random.randint(-100, 100)
    while True:
        b = np.random.randint(-100, 100)
        if b != 0:
            break

    operation, func = random.choice(operations)
    c = func(a, b)

    return f"{a}{operation}{b}={c}"


def main() -> None:
    parser = argparse.ArgumentParser(description="dataset maker")
    parser.add_argument("-n", "--dataset-size", type=int, default=10000,
                        metavar="N", help="size of dataset (default: 10000)")
    parser.add_argument("-o", "--output", type=str, default="dataset.csv", help="file name of output")
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    with open(args.output, "w") as f:
        writer = csv.writer(f)
        for _ in tqdm(range(args.dataset_size)):
            data = make_data()
            writer.writerow(data)


if __name__ == "__main__":
    main()
