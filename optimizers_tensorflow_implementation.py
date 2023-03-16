import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import imageio
import PIL.Image as Image

def rosenbrock(x1, x2):
    return (1-x1)**2 + 100*(x2-x1**2)**2 

# Function to visualize a 3D plot
def plot_3d(x1_, x2_, z, x1, x2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1[0], x2[0], rosenbrock(x1[0], x2[0]), c='r', marker='o', label='SGD')
    ax.scatter(x1[1], x2[1], rosenbrock(x1[1], x2[1]), c='b', marker='o', label='Adagrad')
    ax.scatter(x1[2], x2[2], rosenbrock(x1[2], x2[2]), c='g', marker='o', label='Adam')
    ax.plot_surface(x1_, x2_, z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('z')
    ax.set_title('Rosenbrock function')
    ax.legend()
    plt.show(block=False)
    plt.pause(0.1) # Pause for interval seconds.
    plt.close('all')
    return fig

# Function to convert a figure to an image
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return Image.frombytes("RGBA", (w, h), buf.tostring())

# SGD optimizer
class SGD():
  def __init__(self, lr=0.0000001):
    self.lr = lr


  def calculate_gradient(self, x1, x2):

    grads_x1 = 400*x1**3 - 400*x1*x2 + 2*x1 - 2
    grads_x2 =  200*(x2-x1**2)
    return grads_x1, grads_x2


  def apply_gradient(self, grads, x1, x2):

    x1.assign(x1 - self.lr * grads[0])
    x2.assign(x2 - self.lr * grads[1])


# Adagrad optimizer
class AdaGrad():
  def __init__(self, lr=2, epsilon=1e-7):

    self.lr = lr
    self.epsilon = epsilon
    self.accumulator = (0.1, 0.1)

    
  def calculate_gradient(self, x1, x2):

    grads_x1 = 400*x1**3 - 400*x1*x2 + 2*x1 - 2
    grads_x2 =  200*(x2-x1**2)

    self.accumulator = (self.accumulator[0] + grads_x1**2, self.accumulator[1] + grads_x2**2)
    return grads_x1, grads_x2


  def apply_gradient(self, grads, x1, x2):
 
    x1.assign(x1 - self.lr * grads[0] / (self.accumulator[0] + self.epsilon)**0.5)
    x2.assign(x2 - self.lr * grads[1] / (self.accumulator[1] + self.epsilon)**0.5)

# Adam optimizer
class Adam():
  def __init__(self, lr=0.5, beta1=0.9, beta2=0.999, epsilon=1e-7):

    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.v = [0, 0]
    self.m = [0, 0]
    self.t = 0
    
  def calculate_gradient(self, x1, x2):

    self.t += 1
    grads_x1 = 400*x1**3 - 400*x1*x2 + 2*x1 - 2
    grads_x2 = 200*(x2-x1**2)

    self.m = (self.beta1 * self.m[0] + (1 - self.beta1) * grads_x1, self.beta1 * self.m[1] + (1 - self.beta1) * grads_x2)
    self.v = (self.beta2 * self.v[0] + (1 - self.beta2) * grads_x1**2, self.beta2 * self.v[1] + (1 - self.beta2) * grads_x2**2)

    m_hat = (self.m[0] / (1 - self.beta1**self.t), self.m[1] / (1 - self.beta1**self.t))
    v_hat = (self.v[0] / (1 - self.beta2**self.t), self.v[1] / (1 - self.beta2**self.t))

    return m_hat, v_hat

  def apply_gradient(self, grads, x1, x2):

    x1.assign(x1 - self.lr * grads[0][0] / (self.v[0] + self.epsilon)**0.5)
    x2.assign(x2 - self.lr * grads[0][1] / (self.v[1] + self.epsilon)**0.5)


# Evaluation
#deine a tf tensor
x1_sgd = tf.Variable(-20.0)
x2_sgd = tf.Variable(-0.0)
x1_adagrad = tf.Variable(-20.0)
x2_adagrad = tf.Variable(-0.0)
x1_adam = tf.Variable(-20.0)
x2_adam = tf.Variable(-0.0)


opt_1 = SGD()
opt_2 = AdaGrad()
opt_3 = Adam()


epochs = 50
images = []

for i in range(epochs):
  x = [x1_sgd, x1_adagrad, x1_adam]
  y = [x2_sgd, x2_adagrad, x2_adam]
  x1_ = np.arange(-20, 20, 1)
  x2_ = np.arange(-20, 100, 1)
  x1_, x2_ = np.meshgrid(x1_, x2_)
  z = rosenbrock(x1_, x2_)
  fig = plot_3d(x1_, x2_, z, x, y)
  images.append(fig2img(fig))
  grads = opt_1.calculate_gradient(x1_sgd, x2_sgd)
  grads_adagrad = opt_2.calculate_gradient(x1_adagrad, x2_adagrad)
  grads_adam = opt_3.calculate_gradient(x1_adam, x2_adam)

  opt_1.apply_gradient(grads, x1_sgd, x2_sgd)
  opt_2.apply_gradient(grads_adagrad, x1_adagrad, x2_adagrad)
  opt_3.apply_gradient(grads_adam, x1_adam, x2_adam)

# save the images as a gif
imageio.mimsave('rosenbrock.gif', images, duration=0.1)