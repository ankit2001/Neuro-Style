


import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import pprint
get_ipython().magic('matplotlib inline')


# ## 1 - Problem Statement


pp = pprint.PrettyPrinter(indent=4)
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
pp.pprint(model)


# get $J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$.
# 
# ### 3.1 - Computing the content cost
# 
# In our running example, the content image C will be the picture of the Louvre Museum in Paris. Run the code below to see a picture of the Louvre.

# In[ ]:

content_image = scipy.misc.imread("images/louvre.jpg")
imshow(content_image);



def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = None
    
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = None
    a_G_unrolled = None
    
    # compute the cost with tensorflow (≈1 line)
    J_content = None
    ### END CODE HERE ###
    
    return J_content


# In[ ]:

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    print("J_content = " + str(J_content.eval()))




style_image = scipy.misc.imread("images/monet_800600.jpg")
imshow(style_image);


# This was painted in the style of *[impressionism](https://en.wikipedia.org/wiki/Impressionism)*.
# 
# Lets see how you can now define a "style" cost function $J_{style}(S,G)$. 

# ### 3.2.1 - Style matrix
# 
# #### Gram matrix
# * The style matrix is also called a "Gram matrix." 
# * In linear algebra, the Gram matrix G of a set of vectors $(v_{1},\dots ,v_{n})$ is the matrix of dot products, whose entries are ${\displaystyle G_{ij} = v_{i}^T v_{j} = np.dot(v_{i}, v_{j})  }$. 
# * In other words, $G_{ij}$ compares how similar $v_i$ is to $v_j$: If they are highly similar, you would expect them to have a large dot product, and thus for $G_{ij}$ to be large. 
# 
# #### Two meanings of the variable $G$
# * Note that there is an unfortunate collision in the variable names used here. We are following common terminology used in the literature. 
# * $G$ is used to denote the Style matrix (or Gram matrix) 
# * $G$ also denotes the generated image. 
# * For this assignment, we will use $G_{gram}$ to refer to the Gram matrix, and $G$ to denote the generated image.

# 
# #### Compute $G_{gram}$
# In Neural Style Transfer (NST), you can compute the Style matrix by multiplying the "unrolled" filter matrix with its transpose:
# 
# <img src="images/NST_GM.png" style="width:900px;height:300px;">
# 
# $$\mathbf{G}_{gram} = \mathbf{A}_{unrolled} \mathbf{A}_{unrolled}^T$$
# 
# #### $G_{(gram)i,j}$: correlation
# The result is a matrix of dimension $(n_C,n_C)$ where $n_C$ is the number of filters (channels). The value $G_{(gram)i,j}$ measures how similar the activations of filter $i$ are to the activations of filter $j$. 
# 
# #### $G_{(gram),i,i}$: prevalence of patterns or textures
# * The diagonal elements $G_{(gram)ii}$ measure how "active" a filter $i$ is. 
# * For example, suppose filter $i$ is detecting vertical textures in the image. Then $G_{(gram)ii}$ measures how common  vertical textures are in the image as a whole.
# * If $G_{(gram)ii}$ is large, this means that the image has a lot of vertical texture. 
# 
# 
# By capturing the prevalence of different types of features ($G_{(gram)ii}$), as well as how much different features occur together ($G_{(gram)ij}$), the Style matrix $G_{gram}$ measures the style of an image. 

# **Exercise**:
# * Using TensorFlow, implement a function that computes the Gram matrix of a matrix A. 
# * The formula is: The gram matrix of A is $G_A = AA^T$. 
# * You may use these functions: [matmul](https://www.tensorflow.org/api_docs/python/tf/matmul) and [transpose](https://www.tensorflow.org/api_docs/python/tf/transpose).

# In[ ]:

# GRADED FUNCTION: gram_matrix

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    ### START CODE HERE ### (≈1 line)
    GA = None
    ### END CODE HERE ###
    
    return GA


# In[ ]:

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2*1], mean=1, stddev=4)
    GA = gram_matrix(A)
    
    print("GA = \n" + str(GA.eval()))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **GA**
#         </td>
#         <td>
#            [[  6.42230511  -4.42912197  -2.09668207] <br>
#  [ -4.42912197  19.46583748  19.56387138] <br>
#  [ -2.09668207  19.56387138  20.6864624 ]]
#         </td>
#     </tr>
# 
# </table>

# ### 3.2.2 - Style cost

# Your goal will be to minimize the distance between the Gram matrix of the "style" image S and the gram matrix of the "generated" image G. 
# * For now, we are using only a single hidden layer $a^{[l]}$.  
# * The corresponding style cost for this layer is defined as: 
# 
# $$J_{style}^{[l]}(S,G) = \frac{1}{4 \times {n_C}^2 \times (n_H \times n_W)^2} \sum _{i=1}^{n_C}\sum_{j=1}^{n_C}(G^{(S)}_{(gram)i,j} - G^{(G)}_{(gram)i,j})^2\tag{2} $$
# 
# * $G_{gram}^{(S)}$ Gram matrix of the "style" image.
# * $G_{gram}^{(G)}$ Gram matrix of the "generated" image.
# * Remember, this cost is computed using the hidden layer activations for a particular hidden layer in the network $a^{[l]}$
# 

# **Exercise**: Compute the style cost for a single layer. 
# 
# **Instructions**: The 3 steps to implement this function are:
# 1. Retrieve dimensions from the hidden layer activations a_G: 
#     - To retrieve dimensions from a tensor X, use: `X.get_shape().as_list()`
# 2. Unroll the hidden layer activations a_S and a_G into 2D matrices, as explained in the picture above (see the images in the sections "computing the content cost" and "style matrix").
#     - You may use [tf.transpose](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/transpose) and [tf.reshape](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/reshape).
# 3. Compute the Style matrix of the images S and G. (Use the function you had previously written.) 
# 4. Compute the Style cost:
#     - You may find [tf.reduce_sum](https://www.tensorflow.org/api_docs/python/tf/reduce_sum), [tf.square](https://www.tensorflow.org/api_docs/python/tf/square) and [tf.subtract](https://www.tensorflow.org/api_docs/python/tf/subtract) useful.
#     
#     
# #### Additional Hints
# * Since the activation dimensions are $(m, n_H, n_W, n_C)$ whereas the desired unrolled matrix shape is $(n_C, n_H*n_W)$, the order of the filter dimension $n_C$ is changed.  So `tf.transpose` can be used to change the order of the filter dimension.
# * for the product $\mathbf{G}_{gram} = \mathbf{A}_{} \mathbf{A}_{}^T$, you will also need to specify the `perm` parameter for the `tf.transpose` function.

# In[ ]:

# GRADED FUNCTION: compute_layer_style_cost

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = None
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = None
    a_G = None

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = None
    GG = None

    # Computing the loss (≈1 line)
    J_style_layer = None
    
    ### END CODE HERE ###
    
    return J_style_layer


# In[ ]:

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer = compute_layer_style_cost(a_S, a_G)
    
    print("J_style_layer = " + str(J_style_layer.eval()))




STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]



def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style



def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    
    ### START CODE HERE ### (≈1 line)
    J = None
    ### END CODE HERE ###
    
    return J


# In[ ]:

tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()    
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print("J = " + str(J))



tf.reset_default_graph()

sess = tf.InteractiveSession()

content_image = scipy.misc.imread("images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)



style_image = scipy.misc.imread("images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)


generated_image = generate_noise_image(content_image)
imshow(generated_image[0]);



model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")


sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)


a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)


# #### Style cost

# In[ ]:

# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)


# ### Exercise: total cost
# * Now that you have J_content and J_style, compute the total cost J by calling `total_cost()`. 
# * Use `alpha = 10` and `beta = 40`.

# In[ ]:

### START CODE HERE ### (1 line)
J = None
### END CODE HERE ###


# ### Optimizer
# 
# * Use the Adam optimizer to minimize the total cost `J`.
# * Use a learning rate of 2.0.  
# * [Adam Optimizer documentation](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)

# In[ ]:

# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step (1 line)
train_step = optimizer.minimize(J)


# ### Exercise: implement the model
# 
# * Implement the model_nn() function.  
# * The function **initializes** the variables of the tensorflow graph, 
# * **assigns** the input image (initial generated image) as the input of the VGG19 model 
# * and **runs** the `train_step` tensor (it was created in the code above this function) for a large number of steps.
# 
# #### Hints
# * To initialize global variables, use this: 
# ```Python
# sess.run(tf.global_variables_initializer())
# ```
# * Run `sess.run()` to evaluate a variable.
# * [assign](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/assign) can be used like this:
# ```python
# model["input"].assign(image)
# ```
# 

# In[ ]:

def model_nn(sess, input_image, num_iterations = 200):
    
    # Initialize global variables (you need to run the session on the initializer)
    ### START CODE HERE ### (1 line)
    None
    ### END CODE HERE ###
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    ### START CODE HERE ### (1 line)
    None
    ### END CODE HERE ###
    
    for i in range(num_iterations):
    
      
        None
        ### END CODE HERE ###
        
        # Compute the generated image by running the session on the current model['input']
        ### START CODE HERE ### (1 line)
        generated_image = None
        ### END CODE HERE ###

        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image




model_nn(sess, generated_image)


