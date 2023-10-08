import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#et_file = "/home/matthis/Documents/et_improvement.csv"
et_file = "/Users/maillard/Desktop/et_hauss.csv"
et = pd.read_csv(et_file, header=None, sep=";")
baseline = et.iloc[1, 1:]
x = et.iloc[0,1:]
plt.axhline(y=0, color="black")
ticks = list(x.astype(int))
ticks[2] = 185
ticks[3] = 215
for i in range(2,et.shape[0]):
      plt.plot(x.astype(int), - et.iloc[i, 1:].astype(float).subtract( baseline.astype(float)), label=et.iloc[i, 0])
plt.xticks(ticks, x.astype(int), rotation=30)
plt.xlabel("Size of training set")
plt.ylabel("Difference with the baseline")
plt.ylim([-4,6])
plt.legend()
plt.show()

"""tc_file = "/home/matthis/Documents/tc_improvement.csv"
tc = pd.read_csv(tc_file, header=None)"""
tc_file = "/Users/maillard/Desktop/tc_hauss.csv"
tc = pd.read_csv(tc_file, header=None, sep=";")
baseline = tc.iloc[1, 1:]
x = tc.iloc[0,1:]
plt.axhline(y=0, color="black")
for i in range(2,tc.shape[0]):
      plt.plot(x.astype(int), -tc.iloc[i, 1:].astype(float).subtract( baseline.astype(float)), label=tc.iloc[i, 0])

plt.ylim([-4,6])
plt.xticks(ticks, x.astype(int), rotation=30)
plt.xlabel("Size of training set")
plt.ylabel("Difference with the baseline")
plt.legend()
plt.show()

"""wt_file = "/home/matthis/Documents/wt_improvement.csv"
wt = pd.read_csv(wt_file, header=None)"""
wt_file = "/Users/maillard/Desktop/wt_hauss.csv"
wt = pd.read_csv(wt_file, header=None, sep=";")
baseline = wt.iloc[1, 1:]
x = wt.iloc[0,1:]
plt.axhline(y=0, color="black")
for i in range(2,wt.shape[0]):
      plt.plot(x.astype(int),-wt.iloc[i, 1:].astype(float).subtract( baseline.astype(float)), label=wt.iloc[i, 0])

plt.ylim([-4,6])
plt.xticks(ticks, x.astype(int), rotation=30)
plt.xlabel("Size of training set")
plt.ylabel("Difference with the baseline")
plt.legend()
plt.show()