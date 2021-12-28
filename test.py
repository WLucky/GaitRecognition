import cv2
import numpy as np
import pdb

T_H = T_W = 64
img = cv2.imread("data/train/001/bg-01/000/001-bg-01-000-001.png")[:, :, 0]


y = img.sum(axis=1)
y_top = (y != 0).argmax(axis=0)
# cumsum: Return the cumulative sum of the elements along a given axis.
y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
img = img[y_top:y_btm + 1, :]
# As the height of a person is larger than the width,
# use the height to calculate resize ratio.
_r = img.shape[1] / img.shape[0]
_t_w = int(T_H * _r)
print(_r, _t_w)
img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)

pdb.set_trace()

print(img.shape)

sum_point = img.sum()
# 先沿着列相加  然后 逐行累加  shape (64,)
sum_column = img.sum(axis=0).cumsum()
x_center = -1
for i in range(sum_column.size):
    if sum_column[i] > sum_point / 2:
        x_center = i
        break

h_T_W = int(T_W / 2)
left = x_center - h_T_W
right = x_center + h_T_W
if left <= 0 or right >= img.shape[1]:
    left += h_T_W
    right += h_T_W
    _ = np.zeros((img.shape[0], h_T_W))
    img = np.concatenate([_, img, _], axis=1)
img = img[:, left:right]

img.astype('uint8')