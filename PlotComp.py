import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy.stats as st

# EulerIter = np.loadtxt('Euler_Iter.txt',delimiter=',');
# EulerTime = np.loadtxt('Euler_Time.txt',delimiter=',');
# EGIter = np.loadtxt('EG_Iter.txt',delimiter=',');
# EGTime = np.loadtxt('EG_Time.txt',delimiter=',');
RKHEIter = np.loadtxt('RKMDA2_Iter.txt', delimiter=',');
RKHETime = np.loadtxt('RKMDA2_Time.txt', delimiter=',');
RKCKIter = np.loadtxt('RKMDA5_Iter.txt', delimiter=',');
RKCKTime = np.loadtxt('RKMDA5_Time.txt', delimiter=',');

# EulerIterAVG = np.mean(EulerIter[:,1:], axis=1);
# EulerTimeAVG = np.mean(EulerTime[:,1:], axis=1);
# EGIterAVG = np.mean(EGIter[:,1:], axis=1);
# EGTimeAVG = np.mean(EGTime[:,1:], axis=1);
RKHEIterAVG = np.mean(RKHEIter[:, 1:], axis=1);
RKHETimeAVG = np.mean(RKHETime[:, 1:], axis=1);
RKCKIterAVG = np.mean(RKCKIter[:, 1:], axis=1);
RKCKTimeAVG = np.mean(RKCKTime[:, 1:], axis=1);

# EulerIterSEM = st.sem(EulerIter[:,1:], axis=1);
# EulerTimeSEM = st.sem(EulerTime[:,1:], axis=1);
# EGIterSEM = st.sem(EGIter[:,1:], axis=1);
# EGTimeSEM = st.sem(EGTime[:,1:], axis=1);
RKHEIterSEM = st.sem(RKHEIter[:, 1:], axis=1);
RKHETimeSEM = st.sem(RKHETime[:, 1:], axis=1);
RKCKIterSEM = st.sem(RKCKIter[:, 1:], axis=1);
RKCKTimeSEM = st.sem(RKCKTime[:, 1:], axis=1);

mpl.rcParams['toolbar'] = 'None';

fig1, ax1 = plt.subplots(1, 2);

fig1size = fig1.get_size_inches();
scale_x = 1.0;
scale_y = 0.8;
fig1.set_size_inches((2 * scale_x * fig1size[0], scale_y * fig1size[1]));

# print(EGIter[:,0].shape);
# print(EGIterAVG.shape);
# print(EGIterSEM.shape);

# ax1[0].errorbar(EulerIter[:,0], EulerIterAVG, fmt='b-',  yerr=EulerIterSEM, ecolor='b', lw=2, label='Euler Iters');
# # ax1[0].errorbar(EGIter[:,0],    EGIterAVG,    fmt='k-.', yerr=EGIterSEM,    ecolor='k', lw=2, label='EG Iters');
# # ax1[0].errorbar(EGIter[0:9,0], EGIterAVG[0:9], fmt='k-.',  yerr=EGIterSEM[0:9], ecolor='k', lw=2, label='EG Iters');
# ax1[0].plot(EGIter[8:10,0], [EGIterAVG[8],100000], 'k-.', lw=2);
ax1[0].errorbar(
    RKHEIter[
        :,
        0],
        RKHEIterAVG,
        fmt='g:',
        yerr=RKHEIterSEM,
        ecolor='g',
        lw=2,
         label='RKHE Iters');
ax1[0].errorbar(
    RKCKIter[
        :,
        0],
        RKCKIterAVG,
        fmt='r--',
        yerr=RKCKIterSEM,
        ecolor='r',
        lw=2,
         label='RKCK Iters');
ax1[0].set_xscale('log');
ax1[0].set_yscale('log');
ax1[0].set_xlabel(r'Problem Size ($N_p + 2N_l$)', fontsize=18, color='k');
ax1[0].set_ylabel('# of Iterations to Convergence', fontsize=18, color='k');

# ax1[1].errorbar(EulerTime[:,0], EulerTimeAVG, fmt='b-',  yerr=EulerTimeSEM, ecolor='b', lw=2, label='Euler Time');
# # ax1[1].errorbar(EGTime[:,0],    EGTimeAVG,    fmt='k-.', yerr=EGTimeSEM,    ecolor='k', lw=2, label='EG Time');
# # ax1[1].errorbar(EGTime[0:9,0], EGTimeAVG[0:9], fmt='k-.',  yerr=EGTimeSEM[0:9], ecolor='k', lw=2, label='EG Time');
# ax1[1].plot(EGTime[8:10,0], EGTimeAVG[8:10], 'k-.', lw=2);
ax1[1].errorbar(
    RKHETime[
        :,
        0],
        RKHETimeAVG,
        fmt='g:',
        yerr=RKHETimeSEM,
        ecolor='g',
        lw=2,
         label='RKHE Time');
ax1[1].errorbar(
    RKCKTime[
        :,
        0],
        RKCKTimeAVG,
        fmt='r--',
        yerr=RKCKTimeSEM,
        ecolor='r',
        lw=2,
         label='RKCK Time');
ax1[1].set_xscale('log');
ax1[1].set_yscale('log');
ax1[1].set_xlabel(r'Problem Size ($N_p + 2N_l$)', fontsize=18, color='k'););
ax1[1].set_ylabel('Time to Completion (sec)', fontsize = 18, color = 'k');

# Now add the legend with some customizations.
legend1=ax1[0].legend(loc = 'lower right', shadow = True);
legend2=ax1[1].legend(loc = 'lower right', shadow = True);

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame1=legend1.get_frame();
frame1.set_facecolor('0.90');
frame2=legend2.get_frame();
frame2.set_facecolor('0.90');

# Set the fontsize
for label in legend1.get_texts():
    label.set_fontsize('large');
for label in legend2.get_texts():
    label.set_fontsize('large');

for label in legend1.get_lines():
    label.set_linewidth(1.5);  # the legend line width
for label in legend2.get_lines():
    label.set_linewidth(1.5);  # the legend line width

fig1.suptitle('SCN Convergence to Equilibrium', fontsize = 20);

fig1.savefig('SCNwError.png', bbox_inches = 'tight');
plt.show();



