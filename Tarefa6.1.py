import time
import threading
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.animation as anim

from numpy import array, random, dot, ones, hstack, amax, exp, sqrt, asscalar
from numpy.linalg import linalg

MAX_ITER = 1000000

matplotlib.use("TkAgg")
# style.use('fivethirtyeight')

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
        last_x = xs[len(xs)-1]
        last_y = ys[len(ys)-1]
        xs.clear()
        ys.clear()
        xs.append(last_x)
        ys.append(last_y)
    lock.release()


an = anim.FuncAnimation(fig, animate, interval=20, blit=False)
plot.show(block=False)


def mag(x):
    return asscalar(sqrt(dot(x.T, x)))


def norm(x):
    return x/mag(x)


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


random.seed(int(time.time()))


def training():
    x_p = array(([[1, .1, .1],
                  [1, .1, .9],
                  [1, .9, .1],
                  [1, .9, .9]]))
    # x_p = array(([[0, 0],
    #               [0, 1],
    #               [1, 0],
    #               [1, 1]]))

    y_p = array([[0, 1, 1, 0]]).T

    w_l1 = 2*random.random((3, 3))-1
    w_l2 = 2*random.random((3, 1))-1
    # w_l1 = 2*random.random((2, 2))-1
    # w_l2 = 2*random.random((2, 1))-1

    stopped_by_error_threshold = True
    gi = 0

    etha = 0.3
    last_dir = array(([[0, 0, 0, 0]])).T
    for iteration in range(MAX_ITER):
        w_l1 = w_l1                 \
               * array(([[0, 1, 1],
                         [0, 1, 1],
                         [0, 1, 1]])) \
               + array(([[1, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]]))

        u_1 = dot(x_p, w_l1)
        g_1 = sigmoid(u_1)              \
              * array(([[0, 1, 1],
                        [0, 1, 1],
                        [0, 1, 1],
                        [0, 1, 1]]))    \
              + array(([[1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0]]))

        u_2 = dot(g_1, w_l2)
        g_2 = sigmoid(u_2)

        dg_1 = sigmoid_derivative(g_1)
        dg_2 = sigmoid_derivative(g_2)

        e_2 = (y_p - g_2)
        d_2 = e_2 * dg_2
        e_1 = dot(d_2, w_l2.T)

        if mag(e_2) < 0.05:
            print("Final Epoch :")
            print(iteration)
            stopped_by_error_threshold = True
            break

        if iteration % 10000 == 0:
            print("Current Epoch:")
            print(iteration)
            print("Errors: ")
            print(e_1)
            print(e_2)
            print("\nEtha")
            print(etha)

        e_1 = e_1 * dg_1
        e_2 = e_2 * dg_2

        # d_etha = etha * (asscalar(dot(norm(e_2*dg_2).T, last_dir)))
        # etha += d_etha*.001

        d_w_1 = etha * dot(x_p.T, e_1) / 4
        d_w_2 = etha * dot(g_1.T, e_2) / 4

        w_l1 += d_w_1
        w_l2 += d_w_2

        lock.acquire()
        xs.append(gi)
        ys.append(mag(e_2))
        lock.release()

        last_dir = norm(e_2)

        gi += 1

    if not stopped_by_error_threshold:
        print("Iteration :")
        print(MAX_ITER)

    # testing output
    print("\nOutput")
    u_1 = dot(x_p, w_l1)
    g_1 = sigmoid(u_1) \
          * array(([[0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1]])) \
          + array(([[1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0]]))
    u_2 = dot(g_1, w_l2)
    g_2 = sigmoid(u_2)
    print(g_1)
    print(g_2)

    print("\nTraining data")
    print(y_p)

    print("\nError")
    dg_2 = g_2 * (1 - g_2)
    e_2 = (y_p - g_2)
    d_2 = e_2 * dg_2
    e_1 = dot(d_2, w_l2.T)
    print(e_1)
    print(e_2)
    print("\nEtha")
    print(etha)
    print("\nSynaptic Weights")
    print(w_l1 * array(([[0, 1, 1],
                         [0, 1, 1],
                         [0, 1, 1]])) \
               + array(([[1, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]])))
    print(w_l2)
    # an.event_source.stop()


t = threading.Thread(target=training)
t.start()
