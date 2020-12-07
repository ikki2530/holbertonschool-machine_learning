#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# main stuff
persons = ["Farrah", "Fred", "Felicia"]
width = 0.5
plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
# bars
plt.bar(persons, fruit[0], width, color='r', label="apples")
plt.bar(persons, fruit[1], width, bottom=fruit[0], color='yellow',
        label="bananas")
plt.bar(persons, fruit[2], width, bottom=fruit[1] + fruit[0],
        color='#ff8000', label="oranges")
plt.bar(persons, fruit[3], width, bottom=fruit[2] + fruit[1] + fruit[0],
        color='#ffe5b4', label="peaches")

plt.yticks(np.arange(0, 81, 10))
plt.legend(loc="upper right")
plt.show()
