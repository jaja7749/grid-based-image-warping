import numpy as np

class resize:
    """
    The resize class is used to resize the image using bicubic interpolation.
    
    Parameters
    ------------
    image: np.array (height, width, channel)
            Input image.
    new_height: int
            The height of the output image.
    new_width: int
            The width of the output image.
            
    Returns
    ------------
    resized_image: np.array (new_height, new_width, channel)
            The output image after resizing.
            
    Notes
    ------------
    1. The bicubic interpolation is used to calculate the pixel value of the output image.
    2. If the input image is a color image, the output image will be a color image.
    3. It may have white boundary or black region in the output image.
    
    Examples
    ------------
    >>> import cv2
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    
    >>> image = cv2.imread('image.png')
    >>> resizer = resize()
    >>> new_image = resizer(image, 700, 700)
    >>> print("original image shape: ", image.shape)
    original image shape:  (500, 500, 3)

    >>> print("new image shape: ", new_image.shape)
    new image shape:  (700, 700, 3)
    
    References
    ------------
    Bicubic Interpolation:
    R. Keys, "Cubic convolution interpolation for digital image processing,"
    IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 29, no. 6, pp. 1153-1160, December 1981, 
    doi: 10.1109/TASSP.1981.1163711
    """

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

    def bicubic_interpolation(self, image, x, y):
        """
        Calculate the new pixel value of the output image by bicubic interpolation.
        
        Bicubic interpolation for new value:
            f(x) = \sum img_{i, j} * W(\left \lfloor x_{p}\right \rfloor-x_{i}) *
            W({\left \lfloor y_{q}\right \rfloor-y_{j}}),
            where \  i = [p-1, p, p+1, p+2], j = [q-1, q, q+1, q+2]
        """
        image_shape = image.shape
        channel = image.shape[2]
        height, width = x.shape, y.shape
        length = height[0] * width[0]
        x0 = np.floor(x)
        y0 = np.floor(y)
        
        # get the original region, img_{i, j}, where i = [p-1, p, p+1, p+2], j = [q-1, q, q+1, q+2]
        original_point_x, original_point_y = np.meshgrid(x0, y0, indexing='ij')
        original_point_x, original_point_y = original_point_x.flatten(), original_point_y.flatten()
        original_point = np.stack((original_point_x, original_point_y), axis=1)
        original_region = np.zeros((original_point.shape[0], original_point.shape[1], 16), dtype=np.int32)
        
        for i in range(4):
            for j in range(4):
                original_region[:, 0, i*4+j] = original_point[:, 0] + (i-1)
                original_region[:, 1, i*4+j] = original_point[:, 1] + (j-1)
        
        # clip the region to the image boundary
        original_region[:, 0] = np.clip(original_region[:, 0], 0, image_shape[0]-1)
        original_region[:, 1] = np.clip(original_region[:, 1], 0, image_shape[1]-1)
        
        # crop the region from the image
        image_crop = np.zeros((length, 16, channel))
        
        for i in range(16):
            image_crop[:, i] = image[original_region[:, 0, i], original_region[:, 1, i]]
        
        # calculate the weight
        XX, YY = np.meshgrid(x, y, indexing='ij')
        XX, YY = XX.flatten(), YY.flatten()
        XXYY = np.stack((XX, YY), axis=1)
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
        result = np.reshape(result, (height[0], width[0], channel))
        return result

    def resize_image_bicubic(self, image, new_height, new_width):
        image = np.array(image).astype(np.float32)
        old_height, old_width = image.shape[:2]
        
        # calculate the scale
        scale_x = old_height / new_height
        scale_y = old_width / new_width
        
        # calculate the new coordinate
        new_x = np.arange(new_height) * scale_x
        new_y = np.arange(new_width) * scale_y

        # bicubic interpolation
        resized_image = self.bicubic_interpolation(image, new_x, new_y)
                
        return resized_image.astype(np.uint8)
    
    def __call__(self, image, new_height, new_width):
        return self.resize_image_bicubic(image, new_height, new_width)
    


class resize_original_image:
    """
    The resize class is used to resize the image using bicubic interpolation.
    If the input image is crop from the original image, you can supply 
    the original image and the local point to avoid black region or white boundary.
    
    Parameters
    ------------
    image: np.array (height, width, channel)
            Input image.
    original_image: np.array (height, width, channel)
            The original image.
    local_point: tuple (x, y)
            The local point of the image in the original image.
    new_height: int
            The height of the output image.
    new_width: int
            The width of the output image.
            
    Returns
    ------------
    resized_image: np.array (new_height, new_width, channel)
            The output image after resizing.
            
    Notes
    ------------
    1. The bicubic interpolation is used to calculate the pixel value of the output image.
    2. If the input image is a color image, the output image will be a color image.
    3. If you don't need to crop the image, you can set image = original_image and  the local_point to (0, 0).
    
    Examples
    ------------
    >>> import cv2
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    
    >>> image = cv2.imread('image.png')
    >>> resizer = resize_original_image()
    >>> new_image = resizer(image[100:200, 100:200], image, (100, 100), 200, 200)
    >>> print("original image shape: ", image[100:200, 100:200].shape)
    original image shape:  (100, 100, 3)

    >>> print("new image shape: ", new_image.shape)
    new image shape:  (200, 200, 3)
    
    >>> plt.subplot(121)
    >>> plt.imshow(image)
    >>> plt.subplot(122)
    >>> plt.imshow(new_image)
    >>> plt.show()
    
    References
    ------------
    Bicubic Interpolation:
    R. Keys, "Cubic convolution interpolation for digital image processing,"
    IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 29, no. 6, pp. 1153-1160, December 1981, 
    doi: 10.1109/TASSP.1981.1163711
    """

    def cubic_weight(self, x, a=-0.5):
        """
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

    def bicubic_interpolation(self, image, original_image, local_point, x, y):
        """
        Bicubic interpolation for new value:
        f(x) = \sum img_{i, j} * W(\left \lfloor x_{p}\right \rfloor-x_{i}) * W({\left \lfloor y_{q}\right \rfloor-y_{j}}), where \  i = [p-1, p, p+1, p+2], j = [q-1, q, q+1, q+2]
        """
        image_shape = image.shape
        channel = image.shape[2]
        height, width = x.shape, y.shape
        length = height[0] * width[0]
        x0 = np.floor(x)
        y0 = np.floor(y)
        
        # get the original region, img_{i, j}, where i = [p-1, p, p+1, p+2], j = [q-1, q, q+1, q+2]
        original_point_x, original_point_y = np.meshgrid(x0, y0, indexing='ij')
        original_point_x, original_point_y = original_point_x.flatten(), original_point_y.flatten()
        original_point = np.stack((original_point_x, original_point_y), axis=1)
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
        XX, YY = np.meshgrid(x, y, indexing='ij')
        XX, YY = XX.flatten(), YY.flatten()
        XXYY = np.stack((XX, YY), axis=1)
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
        result = np.reshape(result, (height[0], width[0], channel))
        return result

    def resize_image_bicubic(self, image, original_image, local_point, new_height, new_width):
        image = np.array(image).astype(np.float32)
        original_image = np.array(original_image).astype(np.float32)
        old_height, old_width = image.shape[:2]
        
        # calculate the scale
        scale_x = old_height / new_height
        scale_y = old_width / new_width
        
        # calculate the new coordinate
        new_x = np.arange(new_height) * scale_x
        new_y = np.arange(new_width) * scale_y

        # bicubic interpolation
        resized_image = self.bicubic_interpolation(image, original_image, local_point, new_x, new_y)
        return resized_image.astype(np.uint8)
    
    def __call__(self, image, original_image, local_point, new_height, new_width):
        return self.resize_image_bicubic(image, original_image, local_point, new_height, new_width)