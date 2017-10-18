import pandas as pd
from pylab import *
import tsne
import re
from sklearn.manifold import TSNE


y = pd.read_excel('label_new.xlsx', names=range(8)).T.apply(argmax).values
data = pd.read_excel('数据（1000组）.xlsx', index_col=0, skiprows=4).iloc[:1000, :]


def transfer(x):
    if type(x) == str:
        w = re.findall('[\d\.]+', x)
        if not w:
            return float(w)
        else:
            return 1
    else:
        return x


x = data.fillna(0).iloc[:, :-2].apply(transfer)
t = TSNE(3)
w = t.fit_transform(x, y)

w.shape

tsne.plot_embedding(w[1:, (0, 1)], y, '')
ww = w[1:, (0, 1)]
figure()

ax = subplot(221)
q = y == 0
ax.set_xticks([])
ax.set_yticks([])
plot(ww[q, 0], ww[q, 1], 'r.', markersize=1)
axis([-150, 150] * 2)
title('feature 0,5; label 0')
ax = subplot(222)
q = y == 1
ax.set_xticks([])
ax.set_yticks([])
plot(ww[q, 0], ww[q, 1], 'c.', markersize=5)
axis([-150, 150] * 2)
title('feature 0,5; label 1')
ax = subplot(223)
q = y == 5
ax.set_xticks([])
ax.set_yticks([])
plot(ww[q, 0], ww[q, 1], 'm.', markersize=5)
axis([-150, 150] * 2)
title('feature 0,5; label 5')
ax = subplot(224)
q = y == 6
ax.set_xticks([])
ax.set_yticks([])
plot(ww[q, 0], ww[q, 1], 'y.', markersize=5)
axis([-150, 150] * 2)
title('feature 0,5; label 6')

show()
