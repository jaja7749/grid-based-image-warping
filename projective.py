import numpy as np

class projective_transform:
    """
    This transformation, also known as a perspective transform or homography, operates on homogeneous coordinates.
    Warp the image using the perspective transformation matrix H.
    
    Parameters
    ------------
    image: np.array (height, width, channel)
            Input image.
    original_image: np.array (height, width, channel)
            Original image (avoid the image after cropping).
    src: np.array [[x0, y0], [x1, y1] ... [xn, yn]] 
            The source points.
    dst: np.array [[x0, y0], [x1, y1] ... [xn, yn]] 
            the destination points.
    dsize: tuple (height, width)
            Size of the output image.
    local_point: tuple (x, y) 
            the local point of the image in the original image.
    
    Returns
    ------------
    warp_image: np.array (height, width, channel)
            The output image after the perspective transformation.
            
    Notes
    ------------
    1. The perspective transformation matrix H is calculated by solving the linear equation.
    2. If the input image is a color image, the output image will be a color image.
    3. If you don't need to crop the image, you can set image = original_image and  the local_point to (0, 0).
    4. The bicubic interpolation is used to calculate the pixel value of the output image.
    
    Examples
    ------------
    >>> import cv2
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    
    >>> image = cv2.imread('image.png')
    >>> src = np.array([[0, 0], [0, 500], [500, 500], [500, 0]])
    >>> dst = np.array([[0, 0], [0, 500], [510, 510], [500, 0]])
    >>> dsize = (500, 500)
    >>> local_point = (0, 0)
    >>> warp = Projection(image, image, src, dst, dsize, local_point)
    
    >>> plt.subplot(121)
    >>> plt.imshow(image)
    >>> plt.subplot(122)
    >>> plt.imshow(warp())
    >>> plt.show()
    
    References
    ------------
    Perspective Transform:
    Szeliski, Richard. Computer vision: algorithms and applications. Springer Nature, 2022.
    2.1 Geometric primitives and transformations (2-20, 2-21) (p.33)
    
    Bicubic Interpolation:
    R. Keys, "Cubic convolution interpolation for digital image processing,"
    IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 29, no. 6, pp. 1153-1160, December 1981, 
    doi: 10.1109/TASSP.1981.1163711
    """
    
    def __init__(self, image, original_image, src, dst, dsize, local_point):
        self.image = np.array(image)
        self.original_image = np.array(original_image)
        self.src = src
        self.dst = dst
        self.dsize = (int(dsize[0]), int(dsize[1]))
        self.local_point = local_point
        self.H = self.perspective_transform_weight(src, dst)
        
    def perspective_transform_weight(self, src, dst):
        """
        calculate the perspective transformation matrix H by solving the linear equation.
        
        The matrix H is a 3x3 matrix.
            H = [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]]
        
        And the matrix H can be calculated by solving the linear equation:
            B = A * H
            where A is 8x8 matrix and B is 8x1 matrix.
            and the matrix H is a 3x3 matrixn, but h33 = 1.
            So, the H would be faltten to 8x1 matrix.
        """
        
        A, B = list(), list()
        x, y = src[:, 0], src[:, 1]
        x_, y_ = dst[:, 0], dst[:, 1]
        
        for i in range(4):
            A.append([x[i], y[i], 1, 0, 0, 0, -x_[i]*x[i], -x_[i]*y[i]])
            A.append([0, 0, 0, x[i], y[i], 1, -y_[i]*x[i], -y_[i]*y[i]])
            B.append(x_[i])
            B.append(y_[i])
        
        A = np.array(A)
        B = np.array(B)
        
        H = np.linalg.solve(A, B)
        H = np.append(H, 1).reshape((3, 3))
        
        return H
    
    def cubic_weight(self, x, a=-0.5):
        """
        Calculate the bicubic interpolation weight.
        
        Bicubic convolution weight function:
            a = -0.5
        
            W(x)=\left\{\begin{matrix} 
            (a+2)\left| x \right|^3 - (a+3) \left|x \right| ^2 + 1 & for \left| x \right| \leq 1,\\ 
            a\left| x \right|^3 - 5a \left|x \right| ^2 + 8a \left|x \right| - 4a & for 1 <  \left| x \right| \leq 2, \\ 
            0 & otherwise, \\ 
            \end{matrix}\right.
        """
        x = np.abs(x)
        f1 = lambda x: ((a+2) * np.power(x, 3) - (a+3) * np.power(x, 2) + 1)
        f2 = lambda x: (a * np.power(x, 3) - 5 * a * np.power(x, 2) + 8 * a * x - 4 * a)
        f3 = lambda x: np.zeros_like(x)
        return np.where(x <= 1, f1(x), np.where(x < 2, f2(x), f3(x)))
    
    def bicubic_interpolation(self, image, original_image, local_point, x, y, dsize):
        """
        Calculate the new pixel value of the output image by bicubic interpolation.
        
        Bicubic interpolation for new value:
            f(x) = \sum img_{i, j} * W(\left \lfloor x_{p}\right \rfloor-x_{i}) *
            W({\left \lfloor y_{q}\right \rfloor-y_{j}}),
            where \  i = [p-1, p, p+1, p+2], j = [q-1, q, q+1, q+2]
        """
        
        channel = image.shape[2]
        height, width = dsize
        length = int(height * width)
        x0 = np.floor(x)
        y0 = np.floor(y)
        
        original_point = np.stack((x0, y0), axis=1)
        original_region = np.zeros((original_point.shape[0], original_point.shape[1], 16), dtype=np.int32)
        
        for i in range(4):
            for j in range(4):
                original_region[:, 0, i*4+j] = original_point[:, 0] + (i-1)
                original_region[:, 1, i*4+j] = original_point[:, 1] + (j-1)
                
        original_region[:, 0] += local_point[0]
        original_region[:, 1] += local_point[1]
        
        # clip the region to the image boundary
        original_region[:, 0] = np.clip(original_region[:, 0], 0, original_image.shape[0]-1)
        original_region[:, 1] = np.clip(original_region[:, 1], 0, original_image.shape[1]-1)
        
        # crop the region from the image
        image_crop = np.zeros((length, 16, channel))
        
        for i in range(16):
            image_crop[:, i] = original_image[original_region[:, 0, i], original_region[:, 1, i]]
        
        # calculate the weight
        XXYY = np.stack((x, y), axis=1)
        XXYY[:, 0] += local_point[0]
        XXYY[:, 1] += local_point[1]
        weight_input = np.zeros(original_region.shape)
        for i in range(16):
            weight_input[:, 0, i] = XXYY[:, 0] - original_region[:, 0, i]
            weight_input[:, 1, i] = XXYY[:, 1] - original_region[:, 1, i]
        weight = self.cubic_weight(weight_input[:, 0]) * self.cubic_weight(weight_input[:, 1])

        # calculate the result
        result = np.zeros((weight.shape[0], 3))
        for i in range(channel):
            temp = (image_crop[: ,:, i] * weight)
            temp = np.sum(temp, axis=1)
            result[:, i] = temp

        # clip the result to [0, 255] and reshape the result
        result = np.clip(result, 0, 255)
        result = np.reshape(result, (height, width, channel))
        return result.astype(np.uint8)
    
    def warp_perspective(self, image, original_image, H, dsize, local_point):
        """
        Calculate the new pixel value of the output image by perspective transformation.
        
        New coordinates can calculate by the following equation:
            [u', v', w'] = H^-1 * [x, y, 1]
            
        Normalize the coordinates:
            u = u'/w', v = v'/w'
        """
        
        dsize = (int(dsize[0]), int(dsize[1]))
        h, w = dsize
        H_inv = np.linalg.inv(H)
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        
        # calculate the new coordinates
        vec = np.array([X.flatten(), Y.flatten(), np.ones(X.size)])
        src_vec = np.dot(H_inv, vec)
        src_vec /= src_vec[2]
        u, v = src_vec[0], src_vec[1]
        
        # bicubic interpolation for the new pixel value
        warp_image = self.bicubic_interpolation(image, original_image, local_point, v, u, dsize)
        
        return warp_image
        
    def __call__(self):
        return self.warp_perspective(self.image, self.original_image, self.H, self.dsize, self.local_point)