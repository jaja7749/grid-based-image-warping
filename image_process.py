import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.transform import PiecewiseAffineTransform, warp

from projective import projective_transform
from resize import resize, resize_original_image

class process:
    """
    The image process class is for warp and scale the image to the same size and same position.
    
    Parameters
    ------------
    image_before: np.array (height, width, channel)
            The image before warp.
    image_after: np.array (height, width, channel)
            The image after warp.
    transfer_file: filename.csv
            The transfer file for the resize point.
    grid: tuple (x, y)
            The grid for the region.
    fix_type: str ["affine", "projective"]
            The fix method for the warp.
            
    Returns
    ------------
    new_before: np.array (height, width, channel)
            The new image before warp.
    new_after: np.array (height, width, channel)
            The new image after warp.
            
    Other functions
    ------------
    check_plot: check the region and resize point and plot the image.
    check_plot2: check the warp grid.
    
    Notes
    ------------
    The transfer file should be made by "image locator".
    The grid should be the same with the grid in "image locator".
    The fix method in projective is will not proform well with overlapping and not continue.
    The grid should have one point in each region in "image locator" .
    
    Example
    ------------
    >>> before = cv2.imread("Before.png")
    >>> after = cv2.imread("After.png")
    >>> before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
    >>> after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

    >>> method = process(before, after, "transfer_file.csv", grid=(7, 7), fix="affine")
    >>> before, after = method()
    
    >>> fig = plt.figure(figsize=(30, 10))
    >>> plt.subplot(1, 3, 1)
    >>> plt.imshow(before)
    >>> plt.subplot(1, 3, 2)
    >>> plt.imshow(after)
    >>> plt.subplot(1, 3, 3)
    >>> plt.imshow(before, alpha=0.5)
    >>> plt.imshow(after, alpha=0.5)
    >>> plt.show()
    
    """
    def __init__(self, image_before, image_after, transfer_file, grid=(7, 7), fix_type="affine"):
        assert fix_type in ["affine", "projective"]
        self.image_before = np.array(image_before)
        self.image_after = np.array(image_after)
        self.grid = grid
        self.fix_type = fix_type
        
        self.grid_x = np.int32(np.linspace(0, image_before.shape[1], grid[0] + 1))
        self.grid_y = np.int32(np.linspace(0, image_before.shape[0], grid[1] + 1))
        
        # organize resize point tables
        data = pd.read_csv(transfer_file, header=0, names=["point", "left", "right"])
        left = data["left"]
        right = data["right"]
        left = left.str.replace("(", "").str.replace(")", "").str.split(", ")
        right = right.str.replace("(", "").str.replace(")", "").str.split(", ")
        before_x = [int(x) for x, y in left]
        before_y = [int(y) for x, y in left]
        after_x = [int(x) for x, y in right]
        after_y = [int(y) for x, y in right]
        self.resize_point = pd.DataFrame({"point": data["point"]-1, "before_x": before_x, "before_y": before_y, "after_x": after_x, "after_y": after_y})
        self.scale_image_and_fill()
        self.locate_region()
        self.calculate_warp()
        self.calculate_warp2()

    def __call__(self):
        
        if self.fix_type == "affine":
            self.fix2()
        elif self.fix_type == "projective":
            self.fix()
            
        return self.local_initial()
        
    
    def scale_image_and_fill(self):
        """
        Make the two images with same zero point and same size with the max and min point,
        and fill the blank with white color.
        """
        resize_fun = resize_original_image()
        
        min_x_point = np.argmin(self.resize_point["before_x"])
        max_x_point = np.argmax(self.resize_point["before_x"])
        min_y_point = np.argmin(self.resize_point["before_y"])
        max_y_point = np.argmax(self.resize_point["before_y"])
        
        #print("The min max x y:", min_x_point, max_x_point, min_y_point, max_y_point)
        
        after_x_min = self.resize_point["after_x"][min_x_point]
        after_y_min = self.resize_point["after_y"][min_y_point]
        after_x_max = self.resize_point["after_x"][max_x_point]
        after_y_max = self.resize_point["after_y"][max_y_point]
        
        before_x_min = self.resize_point["before_x"][min_x_point]
        before_y_min = self.resize_point["before_y"][min_y_point]
        before_x_max = self.resize_point["before_x"][max_x_point]
        before_y_max = self.resize_point["before_y"][max_y_point]
        
        new_width_left = before_x_max - before_x_min
        new_height_left = before_y_max - before_y_min
        new_width_right = after_x_max - after_x_min
        new_height_right = after_y_max - after_y_min
        
        scale_ratio_height = new_height_left / new_height_right
        scale_ratio_width = new_width_left / new_width_right
        
        scale_image = self.image_after.copy()
        original_image = self.image_before.copy()
        
        # axis 0 scale
        scale_region = scale_image[after_y_min:after_y_max + 1]
        scale_region = resize_fun(scale_region, scale_image, (after_y_min, 0), new_height_left, scale_region.shape[1])
        scale_region = np.insert(scale_image[after_y_max + 1:], [0], scale_region, axis=0)
        scale_region_x = np.insert(scale_region, [0], scale_image[0:after_y_min], axis=0)
        
        # axis 1 scale
        scale_region = scale_region_x[:, after_x_min:after_x_max + 1]
        scale_region = resize_fun(scale_region, scale_region_x, (0, after_x_min), scale_region.shape[0], new_width_left)
        scale_region = np.insert(scale_region_x[:, after_x_max + 1:], [0], scale_region, axis=1)
        scale_region_xy = np.insert(scale_region, [0], scale_region_x[:, 0:after_x_min], axis=1)
        
        # recalculate the after point
        self.resize_point["after_x"] = (self.resize_point["after_x"] - after_x_min) * scale_ratio_width + after_x_min
        self.resize_point["after_y"] = (self.resize_point["after_y"] - after_y_min) * scale_ratio_height + after_y_min
        
        # fill the blank and reposition the point
        if after_x_min < before_x_min:
            scale_region_xy = np.insert(scale_region_xy, [0], np.ones((scale_region_xy.shape[0], np.abs(after_x_min - before_x_min), 3), dtype=np.uint8) * 255, axis=1)
            self.resize_point["after_x"] += np.abs(after_x_min - before_x_min)
        else:
            original_image = np.insert(original_image, [0], np.ones((original_image.shape[0], np.abs(before_x_min - after_x_min), 3), dtype=np.uint8) * 255, axis=1)
            self.resize_point["before_x"] += np.abs(after_x_min - before_x_min)
            self.grid_x[1:] += np.abs(after_x_min - before_x_min)
        
        if after_y_min < before_y_min:
            scale_region_xy = np.insert(scale_region_xy, [0], np.ones((np.abs(after_y_min - before_y_min), scale_region_xy.shape[1], 3), dtype=np.uint8) * 255, axis=0)
            self.resize_point["after_y"] += np.abs(after_y_min - before_y_min)
        else:
            original_image = np.insert(original_image, [0], np.ones((np.abs(before_y_min - after_y_min), original_image.shape[1], 3), dtype=np.uint8) * 255, axis=0)
            self.resize_point["before_y"] += np.abs(after_y_min - before_y_min)
            self.grid_y[1:] += np.abs(after_y_min - before_y_min)
            
        if scale_region_xy.shape[0] < original_image.shape[0]:
            self.grid_x[-1] = scale_region_xy.shape[1]
        elif scale_region_xy.shape[1] < original_image.shape[1]:
            self.grid_y[-1] = scale_region_xy.shape[0]
            
        self.image_after = scale_region_xy
        self.image_before = original_image
        
    def locate_region(self):
        """
        Follow the grid to locate the region and assign the region id to the resize point.
        """
        # create the region table
        region_x = (self.grid_x[1:self.grid[0]+1] + self.grid_x[0:self.grid[0]]) / 2
        region_y = (self.grid_y[1:self.grid[1]+1] + self.grid_y[0:self.grid[1]]) / 2
        
        region_x, region_y = np.meshgrid(region_x, region_y)
        region_x, region_y = region_x.flatten(), region_y.flatten()
        self.region = pd.DataFrame({"region_x": region_x, "region_y": region_y, "region_id": np.arange(region_x.shape[0])})
        
        # assign the region id to the resize point
        XX = np.meshgrid(self.resize_point["before_x"], region_x)
        YY = np.meshgrid(self.resize_point["before_y"], region_y)
        X = XX[0] - XX[1]
        Y = YY[0] - YY[1]
        
        distance = np.sqrt(X ** 2 + Y ** 2)
        self.resize_point["region_distance"] = np.min(distance, axis=0)
        self.resize_point["region"] = np.argmin(distance, axis=0)
        
        # check if the region has been assigned to multiple points
        # uni, indices = np.unique(self.resize_point["region"], return_counts=True)
        # if np.max(indices) > 1:
        #     repeat_region = uni[np.argmax(indices)]
        #     raise ValueError(f"The region {repeat_region} has been assigned to multiple points!")
        
    def check_plot(self):
        """
        Check the region and resize point and plot the image.
        """
        x_max = np.max([self.image_before.shape[1], self.image_after.shape[1]])
        y_max = np.max([self.image_before.shape[0], self.image_after.shape[0]])
        
        fig, ax = plt.subplots(1, 3, figsize=(18, 7))
        
        ax[0].imshow(self.image_before)
        ax[0].scatter(self.resize_point["before_x"], self.resize_point["before_y"], c="r", edgecolors="k")
        
        ax[1].imshow(self.image_after)
        ax[1].scatter(self.resize_point["after_x"], self.resize_point["after_y"], c="g", edgecolors="k")
        
        ax[2].scatter(self.region["region_x"], self.region["region_y"], c="purple", s=3, alpha=0.5)
        for idx, item in self.region.iterrows():
            ax[2].text(item["region_x"]+1, item["region_y"]+1, int(item["region_id"]), fontsize=8, color="purple")
        
        for idx, item in self.resize_point.iterrows():
            ax[2].scatter(item["before_x"], item["before_y"], c="r", s=10, alpha=0.5)
            ax[2].scatter(item["after_x"], item["after_y"], c="g", s=10, alpha=0.5)
            x = np.min([item["before_x"], item["after_x"]])
            y = np.min([item["before_y"], item["after_y"]])
            height = np.abs(item["before_y"] - item["after_y"])
            width = np.abs(item["before_x"] - item["after_x"])
            ax[2].text(x+width/2-3, y-1, int(item["point"]), fontsize=8, color="k")
            ax[2].add_patch(Rectangle((x, y), width, height, facecolor="b", alpha=0.4, edgecolor="k"))
            ax[2].plot([item["before_x"], self.region["region_x"][item["region"]]], [item["before_y"], self.region["region_y"][item["region"]]], c="purple", linestyle="dotted", alpha=0.5)
        
        for x in self.grid_x:
            ax[2].axvline(x, color="k", linestyle="--", alpha=0.5)
        for y in self.grid_y:
            ax[2].axhline(y, color="k", linestyle="--", alpha=0.5)
            
        ax[0].set_xlim(0, x_max)
        ax[0].set_ylim(y_max, 0)
        ax[1].set_xlim(0, x_max)
        ax[1].set_ylim(y_max, 0)
        ax[2].set_xlim(0, x_max)
        ax[2].set_ylim(y_max, 0)
            
        #plt.show()
        return fig
                
    def calculate_warp(self):
        """
        Calculate the move height and width, and the warp grid for each region.
        This function is for projective transform.
        
        The example result:
        
        p0----------p3(p0)-------(p3)            p0'---------p3'(p0')---------(p3') 
        |            |             |              |            \                 \  
        |            |             |        ->    |              \                | 
        |            |             |               \               \              | 
        p1----------p2(p1)-------(p2)              p1'-------------p2'(p1')------(p2') 
        
        where region0 -> p0 = (x0, y0) ... p3 = (x3, y3)
        """
        # locate each 4 points for each region
        # 
        #  p0----------p3   p0: (x0, y0)
        #  |            |   p1: (x1, y1)
        #  |            |   p2: (x2, y2)
        #  p1----------p2   p3: (x3, y3)
        #
        self.region.sort_values(['region_id'], inplace=True)
        x = self.grid_x
        y = self.grid_y
        x, y = np.meshgrid(x, y)
        self.region["original_x0"] = np.delete(np.delete(x, -1, axis=0), -1, axis=1).flatten()
        self.region["original_y0"] = np.delete(np.delete(y, -1, axis=0), -1, axis=1).flatten()
        self.region["original_x1"] = np.delete(np.delete(x, 0, axis=0), -1, axis=1).flatten()
        self.region["original_y1"] = np.delete(np.delete(y, 0, axis=0), -1, axis=1).flatten()
        self.region["original_x2"] = np.delete(np.delete(x, 0, axis=0), 0, axis=1).flatten()
        self.region["original_y2"] = np.delete(np.delete(y, 0, axis=0), 0, axis=1).flatten()
        self.region["original_x3"] = np.delete(np.delete(x, -1, axis=0), 0, axis=1).flatten()
        self.region["original_y3"] = np.delete(np.delete(y, -1, axis=0), 0, axis=1).flatten()
        
        # calculate the height and width for each region
        for i in range(4):
            self.region[f"x{i}"] = self.region[f"original_x{i}"]
            self.region[f"y{i}"] = self.region[f"original_y{i}"]
        self.region["width"] = self.region["x2"] - self.region["x0"]
        self.region["height"] = self.region["y2"] - self.region["y0"]
        
        # calculate the move distance for each region
        self.region["move_x"] = self.resize_point.sort_values(['region'])["before_x"].values - self.resize_point.sort_values(['region'])["after_x"].values
        self.region["move_y"] = self.resize_point.sort_values(['region'])["before_y"].values - self.resize_point.sort_values(['region'])["after_y"].values
        
        # calculate the new position for each region after move
        for i in range(4):
            self.region[f"x{i}"] += self.region["move_x"]
            self.region[f"y{i}"] += self.region["move_y"]
        
        self.region[["x0", "x1", "x2", "x3"]] -= np.min(self.region[["x0", "x1", "x2", "x3"]].values)
        self.region[["y0", "y1", "y2", "y3"]] -= np.min(self.region[["y0", "y1", "y2", "y3"]].values)
        
        # record the neighbour region for each region, if the region is on the edge, the neighbour region will be NaN
        neighbour_0 = np.array([self.region["region_id"] - self.grid[0] - 1, self.region["region_id"] - self.grid[0], self.region["region_id"] - 1]).T
        neighbour_1 = np.array([self.region["region_id"] - self.grid[0], self.region["region_id"] - self.grid[0] + 1, self.region["region_id"] + 1]).T
        neighbour_2 = np.array([self.region["region_id"] + 1, self.region["region_id"] + self.grid[0] + 1, self.region["region_id"] + self.grid[0]]).T
        neighbour_3 = np.array([self.region["region_id"] - 1, self.region["region_id"] + self.grid[0] - 1, self.region["region_id"] + self.grid[0]]).T
        
        neighbour_0 = np.where(neighbour_0 < 0, np.NaN, neighbour_0)
        neighbour_1 = np.where(neighbour_1 < 0, np.NaN, neighbour_1)
        neighbour_2 = np.where(neighbour_2 < 0, np.NaN, neighbour_2)
        neighbour_3 = np.where(neighbour_3 < 0, np.NaN, neighbour_3)
        
        neighbour_0 = np.where(neighbour_0 >= self.grid[0]*self.grid[1], np.NaN, neighbour_0)
        neighbour_1 = np.where(neighbour_1 >= self.grid[0]*self.grid[1], np.NaN, neighbour_1)
        neighbour_2 = np.where(neighbour_2 >= self.grid[0]*self.grid[1], np.NaN, neighbour_2)
        neighbour_3 = np.where(neighbour_3 >= self.grid[0]*self.grid[1], np.NaN, neighbour_3)
        
        for i in range(self.grid[1]):
            for j in range(self.grid[0]):
                if j == 0:
                    neighbour_0[i*self.grid[0] + j][0] = np.NaN
                    neighbour_0[i*self.grid[0] + j][2] = np.NaN
                    neighbour_3[i*self.grid[0] + j][:2] = np.NaN
                if j == self.grid[0]-1:
                    neighbour_1[i*self.grid[0] + j][1:] = np.NaN
                    neighbour_2[i*self.grid[0] + j][:2] = np.NaN
        
        # calculate the new position for each region after calculate the average of neighbour region
        # 
        #  p0----------p3      p0': (x0_, y0_)
        #   \           \      p1': (x1_, y1_)
        #    \           \     p2': (x2_, y2_)
        #    p1-----------p2   p3': (x3_, y3_)
        #
        x0_ = np.array(self.region["x0"])
        y0_ = np.array(self.region["y0"])
        for i in range(self.grid[0]*self.grid[1]):
            if not(np.isnan(neighbour_0[i][0])):
                x0_[i] += self.region["x2"][int(neighbour_0[i][0])]
                y0_[i] += self.region["y2"][int(neighbour_0[i][0])]
            if not(np.isnan(neighbour_0[i][1])):
                x0_[i] += self.region["x1"][int(neighbour_0[i][1])]
                y0_[i] += self.region["y1"][int(neighbour_0[i][1])]
            if not(np.isnan(neighbour_0[i][2])):
                x0_[i] += self.region["x3"][int(neighbour_0[i][2])]
                y0_[i] += self.region["y3"][int(neighbour_0[i][2])]

            x0_[i] /= (4- np.isnan(neighbour_0[i]).sum())
            y0_[i] /= (4- np.isnan(neighbour_0[i]).sum())
            
        x1_ = np.array(self.region["x1"])
        y1_ = np.array(self.region["y1"])
        for i in range(self.grid[0]*self.grid[1]):
            if not(np.isnan(neighbour_3[i][0])):
                x1_[i] += self.region["x2"][int(neighbour_3[i][0])]
                y1_[i] += self.region["y2"][int(neighbour_3[i][0])]
            if not(np.isnan(neighbour_3[i][1])):
                x1_[i] += self.region["x3"][int(neighbour_3[i][1])]
                y1_[i] += self.region["y3"][int(neighbour_3[i][1])]
            if not(np.isnan(neighbour_3[i][2])):
                x1_[i] += self.region["x0"][int(neighbour_3[i][2])]
                y1_[i] += self.region["y0"][int(neighbour_3[i][2])]

            x1_[i] /= (4- np.isnan(neighbour_3[i]).sum())
            y1_[i] /= (4- np.isnan(neighbour_3[i]).sum())
            
        x2_ = np.array(self.region["x2"])
        y2_ = np.array(self.region["y2"])
        for i in range(self.grid[0]*self.grid[1]):
            if not(np.isnan(neighbour_2[i][0])):
                x2_[i] += self.region["x1"][int(neighbour_2[i][0])]
                y2_[i] += self.region["y1"][int(neighbour_2[i][0])]
            if not(np.isnan(neighbour_2[i][1])):
                x2_[i] += self.region["x0"][int(neighbour_2[i][1])]
                y2_[i] += self.region["y0"][int(neighbour_2[i][1])]
            if not(np.isnan(neighbour_2[i][2])):
                x2_[i] += self.region["x3"][int(neighbour_2[i][2])]
                y2_[i] += self.region["y3"][int(neighbour_2[i][2])]

            x2_[i] /= (4- np.isnan(neighbour_2[i]).sum())
            y2_[i] /= (4- np.isnan(neighbour_2[i]).sum())
            
        x3_ = np.array(self.region["x3"])
        y3_ = np.array(self.region["y3"])
        for i in range(self.grid[0]*self.grid[1]):
            if not(np.isnan(neighbour_1[i][0])):
                x3_[i] += self.region["x2"][int(neighbour_1[i][0])]
                y3_[i] += self.region["y2"][int(neighbour_1[i][0])]
            if not(np.isnan(neighbour_1[i][1])):
                x3_[i] += self.region["x1"][int(neighbour_1[i][1])]
                y3_[i] += self.region["y1"][int(neighbour_1[i][1])]
            if not(np.isnan(neighbour_1[i][2])):
                x3_[i] += self.region["x0"][int(neighbour_1[i][2])]
                y3_[i] += self.region["y0"][int(neighbour_1[i][2])]

            x3_[i] /= (4- np.isnan(neighbour_1[i]).sum())
            y3_[i] /= (4- np.isnan(neighbour_1[i]).sum())
        
        # record the new position to table
        self.region[["x0_", "y0_", "x1_", "y1_", "x2_", "y2_", "x3_", "y3_"]] = np.stack((x0_, y0_, x1_, y1_, x2_, y2_, x3_, y3_), axis=1)
        
    def calculate_warp2(self):
        """
        Calculate the src and dst for warp.
        This function is for skimage. Piecewise Affine Transform.
        
        The example result:
                    src                                       dst
        p0-----------p3------------p4            p0'---------p3'---------------p4' 
        |            |             |              |            \                 \  
        |            |             |        ->    |              \                | 
        |            |             |               \               \              | 
        p1-----------p2------------p5              p1'--------------p2'-----------p5' 
        
        where p0 = (x0, y0) ... pn = (xn, yn)
        """
        src_x, src_y = np.meshgrid(self.grid_x, self.grid_y)
        src = np.vstack([src_x.flat, src_y.flat]).T
        
        dst_x = np.zeros((src_x.shape))
        dst_y = np.zeros((src_y.shape))
        
        dst_x[0:self.grid[1], 0:self.grid[0]] = self.region["x0_"].values.reshape(self.grid[1], self.grid[0])
        dst_y[0:self.grid[1], 0:self.grid[0]] = self.region["y0_"].values.reshape(self.grid[1], self.grid[0])
        dst_x[-1, 0:self.grid[0]] = self.region["x1_"].values.reshape(self.grid[1], self.grid[0])[-1]
        dst_y[-1, 0:self.grid[0]] = self.region["y1_"].values.reshape(self.grid[1], self.grid[0])[-1]
        dst_x[0:self.grid[1], -1] = self.region["x3_"].values.reshape(self.grid[1], self.grid[0])[:, -1]
        dst_y[0:self.grid[1], -1] = self.region["y3_"].values.reshape(self.grid[1], self.grid[0])[:, -1]
        dst_x[-1, -1] = self.region["x2_"].values.reshape(self.grid[1], self.grid[0])[-1, -1]
        dst_y[-1, -1] = self.region["y2_"].values.reshape(self.grid[1], self.grid[0])[-1, -1]
        
        dst = np.vstack([dst_x.flat, dst_y.flat]).T
        
        return src, dst

    def check_plot2(self):
        """
        check the warp grid.
        """
        fig = plt.figure(figsize=(10, 10))
        for i in self.region.iterrows():
            #plt.plot([i[1]["x0"], i[1]["x1"], i[1]["x2"], i[1]["x3"], i[1]["x0"]], [i[1]["y0"], i[1]["y1"], i[1]["y2"], i[1]["y3"], i[1]["y0"]], c="g", alpha=0.2)
            #plt.plot([i[1]["original_x0"], i[1]["original_x1"], i[1]["original_x2"], i[1]["original_x3"], i[1]["original_x0"]], [i[1]["original_y0"], i[1]["original_y1"], i[1]["original_y2"], i[1]["original_y3"], i[1]["original_y0"]], c="r", alpha=0.2)
            plt.plot([i[1]["x0_"], i[1]["x1_"], i[1]["x2_"], i[1]["x3_"], i[1]["x0_"]], [i[1]["y0_"], i[1]["y1_"], i[1]["y2_"], i[1]["y3_"], i[1]["y0_"]], c="b", alpha=0.2)
        #plt.show()
        return fig
           
    def fix(self):
        """
        fix the image with projective transform.
        """
        new_region = list()
        image_crop = list()
        local_point = list()
        src = list()
        dst = list()
        dsize = list()
        
        # calculate the src and dst for each region
        for idx, item in self.region.iterrows():
            image_crop.append(self.image_after[int(item["original_y0"]):int(item["original_y2"]), int(item["original_x0"]):int(item["original_x2"])])
            local_point.append((int(item["original_y0"]), int(item["original_x0"])))
            dsize.append((int(item["height"] + 10), int(item["width"] + 10)))
            
            src_temp = np.array([[item["x0"], item["y0"]], [item["x1"], item["y1"]], [item["x2"], item["y2"]], [item["x3"], item["y3"]]])
            dst_temp = np.array([[item["x0_"], item["y0_"]], [item["x1_"], item["y1_"]], [item["x2_"], item["y2_"]], [item["x3_"], item["y3_"]]])
            min_x = np.min(np.vstack((dst_temp[:, 0], src_temp[:, 0])))
            min_y = np.min(np.vstack((dst_temp[:, 1], src_temp[:, 1])))
            src.append(src_temp - np.array([min_x, min_y]))
            dst.append(dst_temp - np.array([min_x, min_y]))
        
        # warp the image for each region
        for i in range(len(image_crop)):
            projection = projective_transform(image_crop[i], self.image_after, src[i], dst[i], dsize[i], local_point[i])
            new_region.append(projection())
            
        new_image = np.zeros((int(np.max(self.region[["x0_", "x1_", "x2_", "x3_"]])) + 30, int(np.max(self.region[["y0_", "y1_", "y2_", "y3_"]])) + 30, 3), dtype=np.uint8)
        
        # combine the region to the new image based on the p0(x0, y0) of region id
        for idx, item in self.region.iterrows():
            #min_x = np.min([item["x0_"], item["x1_"], item["x2_"], item["x3_"]])
            #min_y = np.min([item["y0_"], item["y1_"], item["y2_"], item["y3_"]])
            new_image[int(item["y0_"]):int(item["y0_"]) + new_region[int(item["region_id"])].shape[0], int(item["x0_"]):int(item["x0_"]) + new_region[int(item["region_id"])].shape[1]] = new_region[int(item["region_id"])]

        self.local_point_after = (self.resize_point["after_x"][0], self.resize_point["after_y"][0])
        
        self.new_after = new_image
    
    def fix2(self):
        """
        fix the image with skimage. Piecewise Affine Transform.
        """
        image = self.image_after.copy()
        src, drc = self.calculate_warp2()

        rows, cols = image.shape[0], image.shape[1]

        tform = PiecewiseAffineTransform()
        tform.estimate(drc, src)
        
        self.local_point_after = tform.inverse((self.resize_point["after_x"][0], self.resize_point["after_y"][0]))
        
        new_after = warp(image, tform, output_shape=(int(rows * 1.05), int(cols * 1.05)))
        self.new_after = (new_after*255).astype(np.uint8)
        
    def local_initial(self):
        """
        locate two image to the same first point
        """
        
        self.local_point_before = (self.resize_point["before_x"][0], self.resize_point["before_y"][0])
        new_before = self.image_before.copy()
        new_after = self.new_after.copy()
        if self.local_point_before[0] < self.local_point_after[0]:
            new_before = np.insert(new_before, [0], np.ones((new_before.shape[0], int(np.abs(self.local_point_after[0] - self.local_point_before[0])), 3), dtype=np.uint8) * 255, axis=1)
            self.local_point_before = (self.local_point_before[0] + int(np.abs(self.local_point_before[0] - self.local_point_after[0])), self.local_point_before[1])
        else:
            new_after = np.insert(new_after, [0], np.ones((new_after.shape[0], int(np.abs(self.local_point_before[0] - self.local_point_after[0])), 3), dtype=np.uint8) * 255, axis=1)
            self.local_point_after = (self.local_point_after[0] + int(np.abs(self.local_point_after[0] - self.local_point_before[0])), self.local_point_after[1])
        if self.local_point_before[1] < self.local_point_after[1]:
            new_before = np.insert(new_before, [0], np.ones((int(np.abs(self.local_point_after[1] - self.local_point_before[1])), new_before.shape[1], 3), dtype=np.uint8) * 255, axis=0)
            self.local_point_before = (self.local_point_before[0], self.local_point_before[1] + int(np.abs(self.local_point_before[1] - self.local_point_after[1])))
        else:
            new_after = np.insert(new_after, [0], np.ones((int(np.abs(self.local_point_before[1] - self.local_point_after[1])), new_after.shape[1], 3), dtype=np.uint8) * 255, axis=0)
            self.local_point_after = (self.local_point_after[0], self.local_point_after[1] + int(np.abs(self.local_point_after[1] - self.local_point_before[1])))
        
        self.new_before = new_before.astype(np.uint8)
        self.new_after = new_after.astype(np.uint8)
        return self.new_before, self.new_after
    