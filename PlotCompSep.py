import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy.stats as st

EulerIter = np.loadtxt('Euler_Iter.txt', delimiter=',')
EulerTime = np.loadtxt('Euler_Time.txt', delimiter=',')
EGIter = np.loadtxt('EG_Iter.txt', delimiter=',')
EGTime = np.loadtxt('EG_Time.txt', delimiter=',')
RKHEIter = np.loadtxt('RKMDA2_Iter.txt', delimiter=',')
RKHETime = np.loadtxt('RKMDA2_Time.txt', delimiter=',')
RKCKIter = np.loadtxt('RKMDA5_Iter.txt', delimiter=',')
RKCKTime = np.loadtxt('RKMDA5_Time.txt', delimiter=',')
ABIter = np.loadtxt('AB_Iter.txt', delimiter=',')
ABTime = np.loadtxt('AB_Time.txt', delimiter=',')

EulerIterAVG = np.mean(EulerIter[:, 1:], axis=1)
EulerTimeAVG = np.mean(EulerTime[:, 1:], axis=1)
EGIterAVG = np.mean(EGIter[:, 1:], axis=1)
EGTimeAVG = np.mean(EGTime[:, 1:], axis=1)
RKHEIterAVG = np.mean(RKHEIter[:, 1:], axis=1)
RKHETimeAVG = np.mean(RKHETime[:, 1:], axis=1)
RKCKIterAVG = np.mean(RKCKIter[:, 1:], axis=1)
RKCKTimeAVG = np.mean(RKCKTime[:, 1:], axis=1)
ABIterAVG = np.mean(ABIter[:, 1:], axis=1)
ABTimeAVG = np.mean(ABTime[:, 1:], axis=1)

EulerIterSEM = st.sem(EulerIter[:, 1:], axis=1)
EulerTimeSEM = st.sem(EulerTime[:, 1:], axis=1)
EGIterSEM = st.sem(EGIter[:, 1:], axis=1)
EGTimeSEM = st.sem(EGTime[:, 1:], axis=1)
RKHEIterSEM = st.sem(RKHEIter[:, 1:], axis=1)
RKHETimeSEM = st.sem(RKHETime[:, 1:], axis=1)
RKCKIterSEM = st.sem(RKCKIter[:, 1:], axis=1)
RKCKTimeSEM = st.sem(RKCKTime[:, 1:], axis=1)
ABIterSEM = st.sem(ABIter[:, 1:], axis=1)
ABTimeSEM = st.sem(ABTime[:, 1:], axis=1)

mpl.rcParams['toolbar'] = 'None'

fig1, ax1 = plt.subplots(1, 1)
fig2, ax2 = plt.subplots(1, 1)

fig1size = fig1.get_size_inches()
scale_x = 1.0
scale_y = 1.0
fig1.set_size_inches((scale_x * fig1size[0], scale_y * fig1size[1]))
fig2.set_size_inches((scale_x * fig1size[0], scale_y * fig1size[1]))

# print(EGIter[:,0].shape);
# print(EGIterAVG.shape);
# print(EGIterSEM.shape);

lw = 5

# ax1.errorbar(EulerIter[:,0], EulerIterAVG, fmt='b-',  yerr=EulerIterSEM, ecolor='b', lw=2, label='Euler Iters');
ax1.errorbar(
    EulerIter[
        0:2,
        0],
    EulerIterAVG[
        0:2],
    fmt='b-',
    yerr=EulerIterSEM[
        0:2],
    ecolor='b',
    lw=lw,
    label='Euler Iters')
# ax1.plot([EulerIter[0,0],EulerIter[1,0],RKHEIter[-1,0]], [EulerIterAVG[0],10.**6,10.**6], 'b-', lw=2);
# ax1.errorbar(EGIter[:,0],    EGIterAVG,    fmt='k-.', yerr=EGIterSEM,    ecolor='k', lw=2, label='EG Iters');
ax1.errorbar(
    EGIter[
        0:2,
        0],
    EGIterAVG[
        0:2],
    fmt='k-.',
    yerr=EGIterSEM[
        0:2],
    ecolor='k',
    lw=lw,
    label='EG Iters')
#ax1.plot([EGIter[0,0],EGIter[1,0],RKHEIter[-1,0]], [EGIterAVG[0],10.**6,10.**6], 'k-.', lw=2);
ax1.errorbar(
    RKHEIter[
        :,
        0],
    RKHEIterAVG,
    fmt='g:',
    yerr=RKHEIterSEM,
    ecolor='g',
    lw=lw,
    label='RKHE Iters')
ax1.errorbar(
    RKCKIter[
        :,
        0],
    RKCKIterAVG,
    fmt='r--',
    yerr=RKCKIterSEM,
    ecolor='r',
    lw=lw,
    label='RKCK Iters')
ax1.errorbar(
    ABIter[
        :,
        0],
    ABIterAVG,
    fmt='c-.',
    yerr=ABIterSEM,
    ecolor='c',
    lw=lw,
    label='AB Iters')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'Problem Size - Dim($X$)', fontsize=20, color='k')
ax1.set_ylabel('# of Iterations to Convergence', fontsize=20, color='k')

# ax2.errorbar(EulerTime[:,0], EulerTimeAVG, fmt='b-',  yerr=EulerTimeSEM, ecolor='b', lw=2, label='Euler Time');
ax2.errorbar(
    EulerTime[
        0:2,
        0],
    EulerTimeAVG[
        0:2],
    fmt='b-',
    yerr=EulerTimeSEM[
        0:2],
    ecolor='b',
    lw=lw,
    label='Euler Time')
#ax2.plot([EulerTime[4,0],RKHETime[5,0],RKHETime[-1,0]], [EulerTimeAVG[4],10.**4,10.**4], 'b-', lw=2);
# ax2.errorbar(EGTime[:,0],    EGTimeAVG,    fmt='k-.', yerr=EGTimeSEM,    ecolor='k', lw=2, label='EG Time');
ax2.errorbar(
    EGTime[
        0:2,
        0],
    EGTimeAVG[
        0:2],
    fmt='k-.',
    yerr=EGTimeSEM[
        0:2],
    ecolor='k',
    lw=lw,
    label='EG Iters')
#ax2.plot([EGTime[3,0],RKHETime[4,0],RKHETime[-1,0]], [EGTimeAVG[3],10.**4,10.**4], 'k-', lw=2);
ax2.errorbar(
    RKHETime[
        :,
        0],
    RKHETimeAVG,
    fmt='g:',
    yerr=RKHETimeSEM,
    ecolor='g',
    lw=lw,
    label='RKHE Time')
ax2.errorbar(
    RKCKTime[
        :,
        0],
    RKCKTimeAVG,
    fmt='r--',
    yerr=RKCKTimeSEM,
    ecolor='r',
    lw=lw,
    label='RKCK Time')
ax2.errorbar(
    ABTime[
        :,
        0],
    ABTimeAVG,
    fmt='c-.',
    yerr=ABTimeSEM,
    ecolor='c',
    lw=lw,
    label='AB Time')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Problem Size - Dim($X$)', fontsize=20, color='k')
ax2.set_ylabel('Time to Completion (sec)', fontsize=20, color='k')

ax1.set_ylim([10. ** 2, 10. ** 6])
ax2.set_ylim([10. ** 0, 10. ** 4])

# Now add the legend with some customizations.
legend1 = ax1.legend(loc='upper right', shadow=True)
legend2 = ax2.legend(loc='lower right', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame1 = legend1.get_frame()
frame1.set_facecolor('0.90')
frame2 = legend2.get_frame()
frame2.set_facecolor('0.90')

# Set the fontsize
for label in legend1.get_texts():
    label.set_fontsize('large')
for label in legend2.get_texts():
    label.set_fontsize('large')

for label in legend1.get_lines():
    label.set_linewidth(1.5)  # the legend line width
for label in legend2.get_lines():
    label.set_linewidth(1.5)  # the legend line width

fig1.suptitle('Blood Banking: Convergence to Equilibrium', fontsize=20)
fig2.suptitle('Blood Banking: Convergence to Equilibrium', fontsize=20)

fig1.savefig('BBwError_Iter.png', bbox_inches='tight')
fig2.savefig('BBwError_Time.png', bbox_inches='tight')
plt.show()

Euler_FTime = np.mean(EulerTime[:, 1:] / EulerIter[:, 1:], (1,))
EG_FTime = np.mean(EGTime[:, 1:] / EGIter[:, 1:] / 2., (1,))
RKHE_FTime = np.mean(RKHETime[:, 1:] / RKHEIter[:, 1:] / 2., (1,))
RKCK_FTime = np.mean(RKCKTime[:, 1:] / RKCKIter[:, 1:] / 6., (1,))
AB_FTime = np.mean(ABTime[:, 1:] / ABIter[:, 1:], (1,))

print(np.array([Euler_FTime, EG_FTime, RKHE_FTime, RKCK_FTime, AB_FTime]))

# eval time increases by 1000 to 1600 times
