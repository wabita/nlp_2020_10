from collections import Counter
import matplotlib.pyplot as plt
import japanize_matplotlib 

with open("pro/94/beam.txt", "r") as f:
    y=[float(line.strip()) for line in f.readlines()]
    x=list(range(1,len(y)+1))

plt.plot(x, y) 
plt.ylabel("BLEUスコア")
plt.xlabel("ビーム幅")
plt.savefig("pro/94/94.png")
plt.show()