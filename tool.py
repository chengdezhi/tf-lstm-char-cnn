kernels         = [ 1,   2,   3,   4,   5,   6,   7]
kernel_features = [50, 100, 150, 200, 200, 200, 200]
sum = 0
for i in range(7):
    sum += kernels[i]*kernel_features[i]
print(sum)
