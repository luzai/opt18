# from lz import *
# source = '/home/opt/ftp/hw1'
# import chardet
# os.chdir('src')
# for fn in os.listdir('.'):
#     # try:
#     #     print(fn)
#     # except:
#     #     print(fn.encode('gb2312'))
#     # print(chardet.detect(fn))
#     match_obj=re.match('.*?(31\d+).*(zip|rar)', fn)
#     if match_obj :
#         # print('true')
#         # print(match_obj.group(1), match_obj.group(2))
#         num,suf=match_obj.group(1), match_obj.group(2)
#         dst = f'{num}.{suf}'
#         if fn!=dst:
#             cp(fn, dst)
#
#
#
from lz import *

for x in (-1, 1):
    for y in (-1, 1):
        for z in (-1, 1):
            if x == -1 and y == 1 and z == 1:
                pass
r = (-1, 1)
x, y, z = np.meshgrid(r, r, r)
x, y, z = x.ravel(), y.ravel(), z.ravel()

c = np.ones(8)
c = ['r'] * 8
c[5] = 'b'
m = ['o'] * 8
m[5] = '^'

c, m

from mpl_toolkits.mplot3d import Axes3D

data = np.vstack((x, y, z)).transpose()
data.shape
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs, ys, zs = data[:, 0], data[:, 1], data[:, 2]
ax.scatter(xs, ys, zs, c=c)
ax.scatter(-1, 1, 1, c='b', marker='^', s=40*4)
# ax.scatter(-1,1,1, c='b', marker='^')
# ax.axis('off')
plt.xticks(np.arange(-1, 1.1, step=0.5))
plt.yticks(np.arange(-1, 1.1, step=0.5))
plt.show()
