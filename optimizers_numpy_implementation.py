import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import imageio
import PIL.Image as Image

# Function for Rosenbrock function
def rosenbrock(x1, x2):
    return (1-x1)**2 + 100*(x2-x1**2)**2 # here x1 and x2 are the variables and a and b are the constants with a=1 and b=100

# Function for Stochastic Gradient Descent
def sgd_rosenbrock(x1, x2, alpha):
    grad_x1 = 400*x1**3 - 400*x1*x2 + 2*x1 - 2
    grad_x2 = 200*(x2-x1**2)
    x1 = x1 - alpha*grad_x1
    x2 = x2 - alpha*grad_x2
    return x1, x2

# Function for Momentum SGD
def momentum_sgd_rosenbrock(x1, x2, alpha, beta, x1_prev, x2_prev):
    x1_prev_temp = x1
    x2_prev_temp = x2
    grad_x1 = 400*x1**3 - 400*x1*x2 + 2*x1 - 2
    grad_x2 = 200*(x2-x1**2)
    x1 = x1 - alpha*grad_x1 + beta*(x1 - x1_prev)
    x2 = x2 - alpha*grad_x2 + beta*(x2 - x2_prev)
    return x1, x2, x1_prev_temp, x2_prev_temp


# Function for Newton's method
def newton_rosenbrock(x1, x2, alpha):
    grad_x1 = 400*x1**3 - 400*x1*x2 + 2*x1 - 2
    grad_x2 = 200*(x2-x1**2)
    hessian = np.array([[1200*x1**2 - 400*x2 + 2, -400*x1], [-400*x1, 200]])
    hessian_inv = np.linalg.inv(hessian)
    grad = np.array([grad_x1, grad_x2])
    grad = grad.reshape(2, 1)
    x = np.array([x1, x2])
    x = x.reshape(2, 1)
    x = x - alpha*np.dot(hessian_inv, grad)

    return x[0][0], x[1][0]

# Function for RMSProp
def rmsprop_rosenbrock(x1, x2,beta, gt_x1_prev, gt_x2_prev, alpha =0.2):
    grad_x1 = 400*x1**3 - 400*x1*x2 + 2*x1 - 2
    grad_x2 = 200*(x2-x1**2)
    gt_x1 = beta*gt_x1_prev + (1-beta)*grad_x1**2
    gt_x2 = beta*gt_x2_prev + (1-beta)*grad_x2**2
    Gt_x1 = np.sqrt(gt_x1)
    Gt_x2 = np.sqrt(gt_x2)

    x1 = x1 - alpha*grad_x1/Gt_x1
    x2 = x2 - alpha*grad_x2/Gt_x2
    return x1, x2, gt_x1, gt_x2

# Function for Adagrad
def adagrad_rosenbrock(x1, x2, grad_sum_x1, grad_sum_x2, alpha =0.2):
    grad_x1 = 400*x1**3 - 400*x1*x2 + 2*x1 - 2
    grad_x2 = 200*(x2-x1**2)
    grad_sum_x1 = grad_sum_x1 + grad_x1**2
    grad_sum_x2 = grad_sum_x2 + grad_x2**2
    x1 = x1 - alpha*grad_x1/np.sqrt(grad_sum_x1)
    x2 = x2 - alpha*grad_x2/np.sqrt(grad_sum_x2)
    return x1, x2, grad_sum_x1, grad_sum_x2



# Function to visualize a 3D plot
def plot_3d(x1_, x2_, z, x1, x2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1[0], x2[0], rosenbrock(x1[0], x2[0]), c='r', marker='o', label='SGD')
    ax.scatter(x1[1], x2[1], rosenbrock(x1[1], x2[1]), c='b', marker='o', label='Momentum SGD')
    ax.scatter(x1[2], x2[2], rosenbrock(x1[2], x2[2]), c='g', marker='o', label='Adagrad')
    ax.scatter(x1[3], x2[3], rosenbrock(x1[3], x2[3]), c='y', marker='o', label='Newton\'s method')
    ax.scatter(x1[4], x2[4], rosenbrock(x1[4], x2[4]), c='m', marker='o', label='RMSProp')
    ax.plot_surface(x1_, x2_, z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.6)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('z')
    ax.set_title('Rosenbrock function')
    ax.legend()
    plt.show(block=False)
    plt.pause(0.0001) # Pause for interval seconds.
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


# running the SGD algorithm for 100 iterations and plotting the trajectory 
x1 = [-20, -20, -20, -20, -20]
x2 = [0, 0, 0, 0, 0]
alpha = 0.0000001
images = []
for i in range(100):
    
    # plot the Rosenbrock function in 3D
    x1_ = np.arange(-20, 20, 0.5)
    x2_ = np.arange(-50, 400, 0.5)
    x1_, x2_ = np.meshgrid(x1_, x2_)
    z = rosenbrock(x1_, x2_)
    fig = plot_3d(x1_, x2_, z, x1, x2)
    # convert the figure to an image
    images.append(fig2img(fig))
    

    # run the SGD algorithm
    x1[0], x2[0] = sgd_rosenbrock(x1[0], x2[0], alpha)

    # run the momentum SGD algorithm
    if i == 0:
        x1_prev = x1[1]
        x2_prev = x2[1]
    x1[1], x2[1], x1_prev, x2_prev = momentum_sgd_rosenbrock(x1[1], x2[1], alpha, 0.9, x1_prev, x2_prev)

    # run Adagrad
    if i == 0:
        grad_x1 = 0
        grad_x2 = 0
    x1[2], x2[2], grad_x1, grad_x2 = adagrad_rosenbrock(x1[2], x2[2], grad_x1, grad_x2, alpha=2)

    # run Newton's method
    x1[3], x2[3] = newton_rosenbrock(x1[3], x2[3], alpha=0.1)

    # run RMSProp
    if i == 0:
        gt_x1_prev = 0
        gt_x2_prev = 0
    x1[4], x2[4], gt_x1_prev, gt_x2_prev = rmsprop_rosenbrock(x1[4], x2[4], 0.9, gt_x1_prev, gt_x2_prev, alpha=0.5)

# save the images as a gif
imageio.mimsave('rosenbrock.gif', images, duration=0.1)


    



