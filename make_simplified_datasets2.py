#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from typing import List

import numpy as np
from tqdm import tqdm


def addition(a: float, b: float) -> float:
    return a + b


def subtraction(a: float, b: float) -> float:
    return a - b


def multiplication(a: float, b: float) -> float:
    return a * b


def division(a: float, b: float) -> float:
    return a / b


def exponentiation(a: float, b: float) -> float:
    return a ** b


def root_extraction(a: float, b: float) -> float:
    return a ** (1/b)


def log(a: float, b: float) -> float:
    return math.log(a, b)


def mod(a: float, b: float) -> float:
    return a % b


operations = [
    ("+", addition),
    ("-", subtraction),
    ("*", multiplication),
    ("/", division),
    ("^", exponentiation),
    ("?", root_extraction),
    ("log", log),
    ("%", mod)
]


def make_data() -> str:
    while True:
        try:
            a = np.random.randn()
            b = np.random.randn()
            operation, func = random.choice(operations)
            c = func(a, b)
            if isinstance(c, float):
                break
        except:
            pass

    return f"{a:.2}{operation}{b:.2}={c:.2}"


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
