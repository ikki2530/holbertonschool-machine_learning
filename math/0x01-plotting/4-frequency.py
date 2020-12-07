#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.title("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")
b = list(range(0, 101, 10))
plt.hist(student_grades, bins=b, edgecolor='black')
plt.xticks(b)
plt.xlim(0, 100)
plt.show()
