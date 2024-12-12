import matplotlib.pyplot as plt
from ..utils.data import generate_deepmind_curves, get_context_set
from ..utils.plotting import plot_functions

context_x, context_y, target_x, target_y = generate_deepmind_curves()

print("Plotting original context")
objidx = 4
plt.scatter(target_x[objidx, :, 0], target_y[objidx, :, 0], label="target")
plt.scatter(context_x[objidx, :, 0], context_y[objidx, :, 0], label="context")
plt.legend()
plt.show()
del (context_x, context_y)

print("Plotting custom sampled context")
context_x, context_y = get_context_set(target_x, target_y)
plt.scatter(target_x[objidx, :, 0], target_y[objidx, :, 0], label="target")
plt.scatter(context_x[objidx, :, 0], context_y[objidx, :, 0], label="context")
plt.legend()
plt.show()
del (context_x, context_y)


dummypred = np.array([target_y.mean() for x in range(target_x.shape[1])])
dummypred = dummypred.reshape(target_y.shape)
dummycontextx, dummycontexty = get_context_set(target_x, target_y)

plot_functions(
    target_x=target_x,
    target_y=target_y,
    context_x=dummycontextx,
    context_y=dummycontexty,
    pred_y=dummypred,
    std=np.zeros_like(dummypred) + 0.2,
)
plt.legend()
plt.title("Test")
del (dummypred, dummycontextx, dummycontexty)
