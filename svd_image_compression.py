# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""

from matplotlib import pyplot as plt
from scipy import linalg
from imageio import imread
import numpy as np

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    #Construct A^HA
    A_herm = A.conj().T.dot(A)

    #Get eigenvals and eigenvecs
    lam, V = linalg.eig(A_herm)
    sigma = np.sqrt(lam)
    #Get sorted indices
    index_list = np.argsort(sigma)[::-1]

    #Square root of sorted evals to get sigma
    sigma, V = sigma[index_list], V[:,index_list]

    #Find indices and count num of nonzero singular vals
    nonzero = sigma > tol
    r = np.sum(nonzero)

    sigma1 = sigma[nonzero]
    V1 = V[:,nonzero]

    #Construct U
    U1 = A.dot(V1) / sigma1

    return U1, sigma1, V1.conj().T

# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    #Set initial values
    theta_range = np.linspace(0, 2*np.pi, 200)

    #Create lists to populate rows of S
    x = np.cos(theta_range)
    y = np.sin(theta_range)

    #Create S and E
    S = np.array([x,y])
    E = np.array([[1,0,0],[0,0,1]])

    #Get SVD of A
    U, s, Vh = linalg.svd(A)
    s = np.diag(s)

    #Create axes to plot transformations
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    #Plot transformations

    #S and E
    ax1.plot(S[0], S[1], linewidth = 2)
    ax1.plot(E[0], E[1], linewidth = 2)
    ax1.axis("equal")

    #VhS and VhE
    VhS = Vh @ S
    VhE = Vh @ E
    ax2.plot(VhS[0], VhS[1], linewidth = 2)
    ax2.plot(VhE[0], VhE[1], linewidth = 2)
    ax2.axis("equal")

    #sVhS and sVhE
    sVhS = s @ Vh @ S
    sVhE = s @ Vh @ E
    ax3.plot(sVhS[0], sVhS[1], linewidth = 2)
    ax3.plot(sVhE[0], sVhE[1], linewidth = 2)
    ax3.axis("equal")

    #UsVhS and UsVhE
    UsVhS = U @ sVhS
    UsVhE = U @ sVhE
    ax4.plot(UsVhS[0], UsVhS[1], linewidth = 2)
    ax4.plot(UsVhE[0], UsVhE[1], linewidth = 2)
    ax3.axis("equal")

    #Show plot
    plt.show()


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    #Check for incorrect size of s
    if s > np.linalg.matrix_rank(A):
        raise ValueError("s is greater than rank of A!")

    #Get SVD
    U, sigma, Vh = linalg.svd(A, full_matrices = False)

    #Get truncated matrices
    U = U[:,:s]
    sigma = sigma[:s]
    Vh = Vh[:s,:]

    #Calculate As
    As = U @ np.diag(sigma) @ Vh

    return As, U.size + sigma.size + Vh.size


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    #Get SVD of A
    U, sigma, Vh = compact_svd(A)

    #Find indices for singular values less that sigma
    indices = np.where(sigma < err)[0]
    if indices.size == 0:
        raise ValueError("A cannot be approximated within the tolerance by a matrix of lesser rank!")

    #Get s from indices
    s = indices[np.argmax(sigma[indices])]

    #Truncate matrices, dropping rows after row s
    U_hat, sigma_hat, Vh_hat = U[:,:s], sigma[:s], Vh[:s,:]

    #Generate s approximation of A
    As = U_hat @ np.diag(sigma_hat) @ Vh_hat

    return As, U_hat.size + sigma_hat.size, Vh_hat.size


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    #Read in image, get its attributes
    image = imread(filename) / 255
    cmap = "gray" if len(image.shape) < 3 else None
    imsize = image.size

    #Compress a grayscale image and get the size
    if cmap == "gray":
        compressed, compsize = svd_approx(image, s)

    #Compress an RGB image
    elif cmap == None:
        #Separate R, G, and B layers
        R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
        #Compress each layer
        Rs, Gs, Bs = svd_approx(R, s), svd_approx(G, s), svd_approx(B, s)
        #Clip each layer to control for variables outside [0,1]
        Rs_cut, Gs_cut, Bs_cut = np.clip(Rs[0], 0, 1), np.clip(Gs[0], 0, 1), np.clip(Bs[0], 0, 1)

        #Concatenate the layers to create full compressed image
        compressed = np.dstack((Rs_cut,Gs_cut,Bs_cut))
        #Get size of compressed image
        compsize = Rs[1] * 3

    #Create subplots to display images
    ax1 = plt.subplot(121)
    plt.axis("off")
    ax2 = plt.subplot(122)
    plt.axis("off")

    #Plot original image and compressed image
    ax1.imshow(image, cmap = cmap)
    ax2.imshow(compressed, cmap = cmap)

    #Title the plot and show it
    plt.suptitle("Difference in entries: " + str(imsize - compsize))
    plt.show()
