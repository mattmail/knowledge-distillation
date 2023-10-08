import numpy as np
import matplotlib.pyplot as plt

#student_path = "../result/student_bneck_2021_kd_t1ce_fold1_0318_1144"
student_path = "../result/student_bneck_2018_kd_t1ce_fold1_quarter_0318_1144"
#baseline_path = "../result/baseline_2021_t1ce_fold1_0317_1043"
baseline_path = "../result/baseline_2018_t1ce_fold1_quarter_0317_1114"


et_s = np.load(student_path +'/dice_et.npy')*100
tc_s = np.load(student_path +'/dice_tc.npy')*100
wt_s = np.load(student_path +'/dice_wt.npy')*100

et_b = np.load(baseline_path +'/dice_et.npy')*100
tc_b = np.load(baseline_path +'/dice_tc.npy')*100
wt_b = np.load(baseline_path +'/dice_wt.npy')*100

diff_et = et_s - et_b
diff_tc = tc_s - tc_b
diff_wt = wt_s - wt_b

size_et = np.load("size_et.npy")
size_tc = np.load("size_tc.npy")
size_wt = np.load("size_wt.npy")


average = (diff_et + diff_tc + diff_wt) / 3
idx = np.argsort(diff_wt.squeeze())
print(diff_et[idx].mean())
print(diff_tc[idx].mean())
print(diff_wt[idx].mean())

print(diff_et[idx][2:].mean())
print(diff_tc[idx][2:].mean())
print(diff_wt[idx][2:].mean())

barWidth = 0.25
idx1 = np.argsort(diff_wt.squeeze())
r1 = np.arange(len(diff_et))
r2 = np.arange(int(len(diff_et)/2))

fig, ax = plt.subplots(2)
idx = np.argsort(size_et[idx1].squeeze())
ax[0].bar(r2, diff_et[idx1][idx].squeeze()[::2], width=1, color="blue", edgecolor='none', label='Dice ET', linewidth=0)
ax[0].legend()
ax[0].set_ylim([-100, 60])
ax[1].bar(r2, size_et[idx1][idx].squeeze()[::2], width=1, color="blue", edgecolor='none', label='Size ET', linewidth=0)
ax[1].legend()
plt.show()

fig, ax = plt.subplots(2)
idx = np.argsort(size_tc[idx1].squeeze())
ax[0].bar(r2, diff_tc[idx1][idx].squeeze()[::2], width=1, color="red", edgecolor='none', label='Dice TC', linewidth=0)
ax[0].legend()
ax[0].set_ylim([-60, 60])
ax[1].bar(r2, size_tc[idx1][idx].squeeze()[::2], width=1, color="red", edgecolor='none', label='Size TC', linewidth=0)
ax[1].legend()
plt.show()

fig, ax = plt.subplots(2)
idx = np.argsort(size_wt[idx1].squeeze())
ax[0].bar(r2, diff_wt[idx1][idx].squeeze()[::2], width=1, color="brown", edgecolor='none', label='Dice WT', linewidth=0)
ax[0].legend()
ax[0].set_ylim([-40, 60])
ax[1].bar(r2, size_wt[idx1][idx].squeeze()[::2], width=1, color="brown", edgecolor='none', label='Size WT', linewidth=0)
ax[1].legend()
plt.show()

"""student_path = "../result/student_bneck_2021_att_t1ce_fold1_0318_1144"
hauss = np.load(student_path+"/hauss_wt.npy")
print(hauss.mean())
print(np.ma.masked_invalid(hauss).mean(), np.ma.masked_invalid(hauss).std())"""