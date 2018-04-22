import threading
import time

import matplotlib
import matplotlib.animation as anim
import matplotlib.pyplot as plot
from numpy import array, random, dot, ones, hstack, exp, sqrt, asscalar, loadtxt, asfarray, genfromtxt, amax, where
from numpy.linalg import linalg

MAX_ITER = 100000000
INPUT = 4
OUTPUT = 1
NEURONS_1ST_LAYER = 4
NEURONS_2ND_LAYER = 2
ETA = 5
ERROR_THRESHOLD = 0.075
PATH = 'dataset/IrisDataSet.txt'

matplotlib.use("TkAgg")
# style.use('fivethirtyeight')

random.seed(int(time.time()))

fig = plot.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []
lock = threading.Lock()

stop_anim = False


def animate(i):
    lock.acquire()
    if len(xs) > 0 and len(ys) > 0:
        ax.plot(xs, ys, 'b-')
        last_x = xs[len(xs) - 1]
        last_y = ys[len(ys) - 1]
        xs.clear()
        ys.clear()
        xs.append(last_x)
        ys.append(last_y)
    lock.release()


an = anim.FuncAnimation(fig, animate, interval=1000, blit=False)
plot.show(block=False)


def mag(x):
    if x.shape[0] == 1:
        return asscalar(sqrt(dot(x, x.T)))
    elif x.shape[1] == 1:
        return asscalar(sqrt(dot(x.T, x)))
    else:
        return linalg.norm(x)

def norm(x):
    return x / mag(x)


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


random.seed(int(time.time()))


def load_dataset(path):
    data = loadtxt(path,
                   dtype=None,
                   comments=';;', delimiter=',')
    data = array(data)
    x = data[:, 0:            INPUT]
    y = data[:, INPUT: (INPUT + OUTPUT)]
    s = data.shape[0]
    return x, y, s


def classify(x):
    return where(x < .4, "Iris-setosa", where(x > .6, "Iris-virginica", "Iris-versicolor"))


def training():
    x, y, n_patterns = load_dataset(PATH)

    x_p = hstack((ones((n_patterns, 1)), x))
    y_p = y

    w_l1 = 2 * random.random((INPUT + 1,             NEURONS_1ST_LAYER + 1)) - 1
    # w_l2 = 2 * random.random((NEURONS_1ST_LAYER + 1, OUTPUT)) - 1
    w_l2 = 2 * random.random((NEURONS_1ST_LAYER + 1, NEURONS_2ND_LAYER + 1)) - 1
    w_l3 = 2 * random.random((NEURONS_2ND_LAYER + 1, OUTPUT)) - 1

    stopped_by_error_threshold = True
    gi = 0

    eta = ETA
    for iteration in range(MAX_ITER):
        w_l1[:, 0] = 0
        w_l1[0][0] = 1

        w_l2[:, 0] = 0
        w_l2[0][0] = 1

        u_1 = dot(x_p, w_l1)
        g_1 = sigmoid(u_1)
        g_1[:, 0] = 1

        u_2 = dot(g_1, w_l2)
        g_2 = sigmoid(u_2)
        g_2[:, 0] = 1

        u_3 = dot(g_2, w_l3)
        g_3 = sigmoid(u_3)

        # u_2 = dot(g_1, w_l2)
        # g_2 = sigmoid(u_2)

        dg_1 = sigmoid_derivative(g_1)
        dg_2 = sigmoid_derivative(g_2)
        dg_3 = sigmoid_derivative(g_3)

        # e_2 = (y_p - g_2)
        # d_2 = e_2 * dg_2
        e_3 = (y_p - g_3)
        d_3 = e_3 * dg_3
        e_2 = dot(d_3, w_l3.T)
        d_2 = e_2 * dg_2
        e_1 = dot(d_2, w_l2.T)

        if amax(abs(e_3)) < ERROR_THRESHOLD:
            print("Final Epoch :")
            print(iteration)
            stopped_by_error_threshold = True
            break

        if iteration % 10000 == 0:
            print("Current Epoch:")
            print(iteration)
            print("Errors: ")
            print(amax(abs(e_1)))
            print(mag(e_1))
            print(amax(abs(e_2)))
            print(mag(e_2))
            print(amax(abs(e_3)))
            print(mag(e_3))
            print("\nEtha")
            print(eta)

        lock.acquire()
        xs.append(gi)
        ys.append(amax(abs(e_3)))
        lock.release()

        gi += 1

        e_1 = e_1 * dg_1
        e_2 = e_2 * dg_2
        e_3 = e_3 * dg_3

        d_w_1 = eta * dot(x_p.T, e_1) / n_patterns
        d_w_2 = eta * dot(g_1.T, e_2) / n_patterns
        d_w_3 = eta * dot(g_2.T, e_3) / n_patterns

        w_l1 += d_w_1
        w_l2 += d_w_2
        w_l3 += d_w_3

    if not stopped_by_error_threshold:
        print("Iteration :")
        print(MAX_ITER)

    # testing output
    print("\nOutput | Training Data")
    u_1 = dot(x_p, w_l1)
    g_1 = sigmoid(u_1)
    g_1[:, 0] = 1
    u_2 = dot(g_1, w_l2)
    g_2 = sigmoid(u_2)
    g_2[:, 0] = 1
    u_3 = dot(g_2, w_l3)
    g_3 = sigmoid(u_3)
    # u_2 = dot(g_1, w_l2)
    # g_2 = sigmoid(u_2)

    print(hstack((classify(y_p), classify(g_3))))

    print("\nError")
    # dg_2 = g_2 * (1 - g_2)
    # e_2 = (y_p - g_2)
    # d_2 = e_2 * dg_2
    dg_3 = g_3 * (1 - g_3)
    e_3 = (y_p - g_3)
    d_3 = e_3 * dg_3
    dg_2 = g_2 * (1 - g_2)
    e_2 = dot(d_3, w_l3.T)
    d_2 = e_2 * dg_2
    e_1 = dot(d_2, w_l2.T)
    print(amax(abs(e_1)))
    print(mag(e_1))
    print(amax(abs(e_2)))
    print(mag(e_2))
    print(amax(abs(e_3)))
    print(mag(e_3))
    print("\nEtha")
    print(eta)
    print("\nSynaptic Weights")
    w_l1[:, 0] = 0
    w_l1[0][0] = 1
    w_l2[:, 0] = 0
    w_l2[0][0] = 1
    print(w_l1)
    print(w_l2)
    print(w_l3)
    # an.event_source.stop()


t = threading.Thread(target=training)
t.start()
