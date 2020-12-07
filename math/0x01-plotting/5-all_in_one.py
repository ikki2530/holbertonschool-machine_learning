#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
# create grid and figure
fig3 = plt.figure(figsize=(20, 11))
gs = fig3.add_gridspec(3, 2)

# 0
f3_ax1 = fig3.add_subplot(gs[0, 0])
# f3_ax1.set_title('gs[0, 0]')
f3_ax1.plot(range(len(y0)), y0, 'r')
f3_ax1.set(xlim=(0, 10))
# 1
f3_ax2 = fig3.add_subplot(gs[0, 1])
f3_ax2.set_title('gs[0, 1]')
f3_ax2.scatter(x1, y1, c="magenta")
f3_ax2.set_title("Men's Height vs Weight")
f3_ax2.set(xlabel="Height (in)", ylabel="Weight (lbs)")
# 2
f3_ax3 = fig3.add_subplot(gs[1, 0])
f3_ax3.set_title('gs[1, 0]')
f3_ax3.plot(x2, y2)
f3_ax3.set(xlabel="Time (years)", ylabel="Fraction Remaining", xlim=(0, 28650),
           title="Exponential Decay of C-14", yscale="log")

# 3
f3_ax4 = fig3.add_subplot(gs[1, 1])
f3_ax4.set_title('Exponential Decay of Radioactive Elements')
f3_ax4.plot(x3, y31, 'r--', label="C-14")
f3_ax4.plot(x3, y32, 'g-', label="Ra-226")
f3_ax4.set(xlabel="Time (years)", ylabel="Fraction Remaining",
           xlim=(0, 20000), ylim=(0, 1))
f3_ax4.legend(loc="upper right")

# 4
f3_ax5 = fig3.add_subplot(gs[2, 0:])
f3_ax5.set_title('gs[1, 0:]')
b = list(range(0, 101, 10))
f3_ax5.set(title="Project A", xlabel="Grades", ylabel="Number of Students",
           xlim=(0, 100), ylim=(0, 30), xticks=b)

f3_ax5.hist(student_grades, bins=b, edgecolor='black')

# fit the spaces between the graphs
fig3.tight_layout()
plt.show()
