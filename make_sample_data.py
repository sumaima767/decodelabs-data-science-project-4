"""Generate a small SYNTHETIC review set so the project runs out of the box.

Replace data/reviews.csv with a real dataset (IMDB, Amazon, Yelp...) for real results.
"""
import random
from pathlib import Path

import pandas as pd

random.seed(7)
NOUNS = ["phone", "blender", "laptop", "headphones", "vacuum", "camera", "jacket", "keyboard", "speaker", "kettle"]
POS = ["great", "excellent", "fantastic", "amazing", "wonderful", "superb", "impressive", "reliable", "lovely", "perfect"]
NEG = ["terrible", "awful", "horrible", "disappointing", "useless", "poor", "dreadful", "flimsy", "annoying", "worthless"]

POS_T = [
    "This {n} is {p}!",
    "I love this {n}, it works {p}ly well and I am very happy.",
    "{P} {n}, highly recommend it to everyone.",
    "The {n} is not {x}, I am really happy with it.",
    "Never been disappointed by this {n}, truly {p}. <br>",
    "Bought the {n} last week and it works perfectly, {p} quality.",
    "I wasn't expecting much but this {n} is {p}.",
]
NEG_T = [
    "This {n} is {x}!!!",
    "I hate this {n}, it broke after two days and the quality is {x}.",
    "{X} {n}, do not buy it.",
    "The {n} is not {p}, I am not happy with it.",
    "It didn't work and I wasn't impressed, {x} {n}. <br>",
    "Returned the {n} to the store, {x} waste of money.",
    "I can't recommend this {n}, it is {x}.",
]


def make(templates, n):
    rows = []
    for _ in range(n):
        t = random.choice(templates)
        p, x = random.choice(POS), random.choice(NEG)
        s = t.format(n=random.choice(NOUNS), p=p, x=x, P=p.capitalize(), X=x.capitalize())
        rows.append(s.replace("greatly well", "really well").replace("perfectly well", "really well"))
    return rows


if __name__ == "__main__":
    df = pd.DataFrame(
        [(t, "positive") for t in make(POS_T, 450)] + [(t, "negative") for t in make(NEG_T, 450)],
        columns=["text", "label"],
    ).sample(frac=1, random_state=7)
    out = Path(__file__).resolve().parents[1] / "data" / "reviews.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")
