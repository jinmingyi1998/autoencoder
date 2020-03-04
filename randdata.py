import numpy as np
from random import random, randint

numbers = np.random.randint(0, 100, [50000, 8])
labels = []
for i, nums in enumerate(numbers):
    m = np.argmax(nums)
    if np.std(nums) > 30:
        mm = 0
        for j in range(8):
            if j == m:
                continue
            if nums[j] > nums[mm]:
                mm = j
        m = mm

    labels.append(m)
