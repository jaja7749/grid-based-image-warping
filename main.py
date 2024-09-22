import tkinter as tk
from tksheet import Sheet
from tkinter import filedialog, ttk, Toplevel
from PIL import Image, ImageTk
import pandas as pd
from image_process import process
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
import numpy as np
import sys

class ImageLocatorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Locator")
        self.master.geometry("1500x600")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.toplevel_windows = []

        # set windows auto resize
        self.master.grid_columnconfigure(0, weight=1) # image left (auto resize)
        self.master.grid_columnconfigure(2, weight=1) # image right (auto resize)
        self.master.grid_columnconfigure(4, weight=0) # table (fixed size)
        self.master.grid_rowconfigure(0, weight=1)    # vertical auto resize

        # initial scrollbar on the left and right image
        self.left_scroll_x = tk.Scrollbar(master, orient=tk.HORIZONTAL)
        self.left_scroll_y = tk.Scrollbar(master, orient=tk.VERTICAL)
        self.right_scroll_x = tk.Scrollbar(master, orient=tk.HORIZONTAL)
        self.right_scroll_y = tk.Scrollbar(master, orient=tk.VERTICAL)

        # initial left and right image canvas
        self.canvas_left = tk.Canvas(master, bg="white", 
                                     xscrollcommand=self.left_scroll_x.set, 
                                     yscrollcommand=self.left_scroll_y.set, 
                                     xscrollincrement=10, 
                                     yscrollincrement=10)
        self.canvas_right = tk.Canvas(master, bg="white", 
                                      xscrollcommand=self.right_scroll_x.set, 
                                      yscrollcommand=self.right_scroll_y.set, 
                                      xscrollincrement=10, 
                                      yscrollincrement=10)
        self.canvas_left.grid(row=0, column=0, sticky="nsew")   # set left canvas position and fill the canvas
        self.canvas_right.grid(row=0, column=2, sticky="nsew")  # set right canvas position and fill the canvas

        # sign scrollbar to image canvas
        self.left_scroll_x.config(command=self.canvas_left.xview)
        self.left_scroll_y.config(command=self.canvas_left.yview)
        self.right_scroll_x.config(command=self.canvas_right.xview)
        self.right_scroll_y.config(command=self.canvas_right.yview)

        # set scrollbar position
        self.left_scroll_x.grid(row=1, column=0, sticky="ew")
        self.left_scroll_y.grid(row=0, column=1, sticky="ns")
        self.right_scroll_x.grid(row=1, column=2, sticky="ew")
        self.right_scroll_y.grid(row=0, column=3, sticky="ns")

        # initial table
        self.sheet = Sheet(master, headers=["Point #", "Left Image (x, y)", "Right Image (x, y)"])
        self.sheet.grid(row=0, column=4, rowspan=2, sticky="nsew") # set table position and fill the table
        self.sheet.set_sheet_data([])
        
        # initial coordinate label
        self.coord_label_left = tk.Label(master, text="Left Image Coordinates: (0, 0)")
        self.coord_label_right = tk.Label(master, text="Right Image Coordinates: (0, 0)")
        self.coord_label_left.grid(row=2, column=0)
        self.coord_label_right.grid(row=2, column=2)

        # initial load image and export csv button
        self.load_button = tk.Button(master, text="Load Images", command=self.load_images)
        self.export_button = tk.Button(master, text="Export CSV", command=self.export_to_csv)
        self.load_button.grid(row=2, column=1)
        self.export_button.grid(row=2, column=4)
        
        # Image Process Controls ==============================================================
        
        # initial process image button
        self.image_process_frame = tk.LabelFrame(master, text="Image Process Controls")
        self.image_process_frame.grid(row=3, column=4, padx=10, pady=10, sticky="nsew")
        
        self.csv_file_label = tk.Label(self.image_process_frame, text="No CSV file selected")
        self.csv_file_label.grid(row=0, column=0, columnspan=4, sticky="w")
        self.load_csv_button = tk.Button(self.image_process_frame, text="Load CSV Filename", command=self.load_csv_file)
        self.load_csv_button.grid(row=1, column=1, padx=5, sticky="ew")
        
        self.image_process_list = ttk.Combobox(self.image_process_frame, values=["affine", "projective"])
        self.image_process_list.current(0)
        self.image_process_list.grid(row=1, column=0, padx=5, sticky="ew")
        self.image_process_check = tk.Button(self.image_process_frame, text="Check Region", command=self.check_process)
        self.image_process_check.grid(row=1, column=2, padx=5, sticky="ew")
        self.image_process_button = tk.Button(self.image_process_frame, text="Process Image", command=self.image_process)
        self.image_process_button.grid(row=1, column=3, padx=5, sticky="ew")
        
        self.image_process_frame.grid_columnconfigure(0, weight=1)
        self.image_process_frame.grid_columnconfigure(1, weight=1)
        self.image_process_frame.grid_columnconfigure(2, weight=1)
        self.image_process_frame.grid_columnconfigure(3, weight=1)
        
        # Zoom Controls ========================================================================
        
        # initial zoom frame, zoom in, and zoom out button and label
        self.zoom_frame = tk.LabelFrame(master, text="Zoom Controls")
        self.zoom_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        
        self.zoom_in_button = tk.Button(self.zoom_frame, text="Zoom In (+)", command=lambda: self.zoom(1.1))
        self.zoom_out_button = tk.Button(self.zoom_frame, text="Zoom Out (-)", command=lambda: self.zoom(0.9))
        self.zoom_label = tk.Label(self.zoom_frame, text="Zoom: 100%")
        
        self.zoom_in_button.grid(row=0, column=0, padx=5, sticky="ew")
        self.zoom_label.grid(row=0, column=1, padx=5, sticky="ew")
        self.zoom_out_button.grid(row=0, column=2, padx=5, sticky="ew")
        
        self.zoom_frame.grid_columnconfigure(0, weight=1)
        self.zoom_frame.grid_columnconfigure(1, weight=1)
        self.zoom_frame.grid_columnconfigure(2, weight=1)
        
        # Grid Controls== ========================================================================
        
        # Add input fields for grid configuration
        self.grid_frame = tk.LabelFrame(master, text="Grid Controls")
        self.grid_frame.grid(row=3, column=1, columnspan=3, padx=10, pady=10, sticky="nsew")
        
        self.grid_col_label = tk.Label(self.grid_frame, text="Grid Columns:")
        self.grid_col_entry = tk.Entry(self.grid_frame)
        self.grid_row_label = tk.Label(self.grid_frame, text="Grid Rows:")
        self.grid_row_entry = tk.Entry(self.grid_frame)

        self.grid_col_label.grid(row=0, column=0)
        self.grid_col_entry.grid(row=0, column=1)
        self.grid_row_label.grid(row=0, column=2)
        self.grid_row_entry.grid(row=0, column=3)

        self.grid_col_entry.insert(0, "7")  # default value
        self.grid_row_entry.insert(0, "7")  # default value

        self.apply_grid_button = tk.Button(self.grid_frame, text="Apply Grid", command=self.update_grid)
        self.apply_grid_button.grid(row=0, column=4)
        
        self.grid_frame.grid_columnconfigure(0, weight=1)
        self.grid_frame.grid_columnconfigure(1, weight=1)
        self.grid_frame.grid_columnconfigure(2, weight=1)
        self.grid_frame.grid_columnconfigure(3, weight=1)
        self.grid_frame.grid_columnconfigure(4, weight=1)

        # Parameters ==========================================================================

        # Parameters to store grid lines
        self.grid_lines_left = []
        self.grid_lines_right = []

        # initial parameters
        self.left_image = None               # left image
        self.right_image = None              # right image
        self.left_coords = []                # left image coordinates
        self.right_coords = []               # right image coordinates
        self.scale_factor = 1.0              # zoom scale factor
        self.left_points = {}                # left image points
        self.right_points = {}               # right image points
        self.left_scaled_points = {}         # Store points scaled to the current zoom level for left image
        self.right_scaled_points = {}        # Store points scaled to the current zoom level for right image
        self.drag_data = {"x": 0., "y": 0.}  # original mouse position
        self.is_dragging = False             # dragging status
        self.csv_file_path = None            # image process csv file path

        # Bind====== ==========================================================================

        # bind left and right image canvas event
        self.canvas_left.bind("<Motion>", self.update_left_coord_label)    # left image mouse move
        self.canvas_right.bind("<Motion>", self.update_right_coord_label)  # right image mouse move
        self.canvas_left.bind("<Button-3>", self.delete_left_point)        # left image right click
        self.canvas_right.bind("<Button-3>", self.delete_right_point)      # right image right click
        
        self.canvas_left.bind("<ButtonPress-1>", self.start_drag_left)     # left image left click
        self.canvas_left.bind("<B1-Motion>", self.drag_left)               # left image drag
        self.canvas_left.bind("<ButtonRelease-1>", self.end_drag_left)     # left image left release

        self.canvas_right.bind("<ButtonPress-1>", self.start_drag_right)   # right image left click
        self.canvas_right.bind("<B1-Motion>", self.drag_right)             # right image drag
        self.canvas_right.bind("<ButtonRelease-1>", self.end_drag_right)   # right image left release

    def load_images(self):
        """load image and point from "load_button" """
        left_image_path = filedialog.askopenfilename(title="Select the original image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        right_image_path = filedialog.askopenfilename(title="Select the transformed image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

        if left_image_path and right_image_path:
            self.original_left_img = Image.open(left_image_path)
            self.original_right_img = Image.open(right_image_path)
            self.update_grid()

            self.display_images()

    def display_images(self):
        """display image on canvas point from "load_images", and "zoom" """
        # calculate the new width and height
        left_width, left_height = int(self.original_left_img.width * self.scale_factor), int(self.original_left_img.height * self.scale_factor)
        right_width, right_height = int(self.original_right_img.width * self.scale_factor), int(self.original_right_img.height * self.scale_factor)
        
        # resize the image
        left_img = self.original_left_img.resize((left_width, left_height), Image.LANCZOS)
        right_img = self.original_right_img.resize((right_width, right_height), Image.LANCZOS)
        
        # update the image
        self.left_image = ImageTk.PhotoImage(left_img)
        self.right_image = ImageTk.PhotoImage(right_img)
        
        # update the canvas
        self.canvas_left.create_image(0, 0, anchor=tk.NW, image=self.left_image)
        self.canvas_right.create_image(0, 0, anchor=tk.NW, image=self.right_image)
        
        # update the scroll region
        self.canvas_left.config(scrollregion=self.canvas_left.bbox(tk.ALL))
        self.canvas_right.config(scrollregion=self.canvas_right.bbox(tk.ALL))
        
        # Draw grid on both canvases
        self.draw_grid(self.canvas_left, self.left_image.width(), self.left_image.height(), self.grid_lines_left)
        self.draw_grid(self.canvas_right, self.right_image.width(), self.right_image.height(), self.grid_lines_right)
        
        # redraw points after zooming
        self.redraw_points(self.canvas_left, self.left_points, self.left_scaled_points)
        self.redraw_points(self.canvas_right, self.right_points, self.right_scaled_points)

        self.update_table()
        
    def update_grid(self):
        """Update the grid when the user changes the column/row input."""
        try:
            self.grid_cols = int(self.grid_col_entry.get())
            self.grid_rows = int(self.grid_row_entry.get())
        except ValueError:
            return  # Handle invalid inputs gracefully

        self.display_images()  # Redraw images with the new grid
        
    def draw_grid(self, canvas, img_width, img_height, grid_lines):
        """Draw a grid on the specified canvas based on image dimensions."""
        for line in grid_lines:
            canvas.delete(line)
        grid_lines.clear()

        if self.grid_cols > 1:
            col_width = img_width / self.grid_cols
            for i in range(1, self.grid_cols):
                x = i * col_width
                grid_lines.append(canvas.create_line(x, 0, x, img_height, fill="gray", dash=(2, 2)))

        if self.grid_rows > 1:
            row_height = img_height / self.grid_rows
            for i in range(1, self.grid_rows):
                y = i * row_height
                grid_lines.append(canvas.create_line(0, y, img_width, y, fill="gray", dash=(2, 2)))


    def mark_left(self, event):
        """mark point on left image canvas point from "start_drag_left" """
        # calculate the actual coordinates considering the scrollbar offset
        x = int((self.canvas_left.canvasx(event.x)) / self.scale_factor)
        y = int((self.canvas_left.canvasy(event.y)) / self.scale_factor)

        # draw the annotation point at the actual click position
        point_id = self.canvas_left.create_oval(self.canvas_left.canvasx(event.x) - 3,  # x0
                                                self.canvas_left.canvasy(event.y) - 3,  # y0
                                                self.canvas_left.canvasx(event.x) + 3,  # x1
                                                self.canvas_left.canvasy(event.y) + 3,  # y1
                                                fill="red", outline="black")
        
        # update the coordinates and the table
        self.left_coords.append((x, y))
        self.left_points[(x, y)] = point_id
        
        # Store the point scaled by the current zoom factor
        self.left_scaled_points[(x, y)] = (self.canvas_left.canvasx(event.x), self.canvas_left.canvasy(event.y))
        
        self.update_table()

    def mark_right(self, event):
        """mark point on right image canvas point from "start_drag_right" """
        # calculate the actual coordinates considering the scrollbar offset
        x = int((self.canvas_right.canvasx(event.x)) / self.scale_factor)
        y = int((self.canvas_right.canvasy(event.y)) / self.scale_factor)

        # draw the annotation point at the actual click position
        point_id = self.canvas_right.create_oval(self.canvas_right.canvasx(event.x) - 3,  # x0
                                                 self.canvas_right.canvasy(event.y) - 3,  # y0
                                                 self.canvas_right.canvasx(event.x) + 3,  # x1
                                                 self.canvas_right.canvasy(event.y) + 3,  # y1
                                                 fill="blue", outline="black")
        
        # update the coordinates and the table
        self.right_coords.append((x, y))
        self.right_points[(x, y)] = point_id
        
        # Store the point scaled by the current zoom factor
        self.right_scaled_points[(x, y)] = (self.canvas_right.canvasx(event.x), self.canvas_right.canvasy(event.y))
        
        self.update_table()

    def delete_left_point(self, event):
        """delete point on left image canvas point from "left <Button-3>" """
        # calculate the actual coordinates considering the scrollbar offset
        x = int((self.canvas_left.canvasx(event.x)) / self.scale_factor)
        y = int((self.canvas_left.canvasy(event.y)) / self.scale_factor)
        closest_point = min(self.left_coords, key=lambda p: (p[0]-x)**2 + (p[1]-y)**2)
        
        # delete the closest point if it is within a certain threshold
        threshold = 100 # adjust the threshold as needed
        if (closest_point[0]-x)**2 + (closest_point[1]-y)**2 < threshold:
            point_id = self.left_points.pop(closest_point, None)
            if point_id:
                self.canvas_left.delete(point_id)
                self.left_coords.remove(closest_point)
                self.update_table()
                self.display_images()

    def delete_right_point(self, event):
        """delete point on right image canvas point from "right <Button-3>" """
        # calculate the actual coordinates considering the scrollbar offset
        x = int((self.canvas_right.canvasx(event.x)) / self.scale_factor)
        y = int((self.canvas_right.canvasy(event.y)) / self.scale_factor)
        closest_point = min(self.right_coords, key=lambda p: (p[0]-x)**2 + (p[1]-y)**2)
        
        # delete the closest point if it is within a certain threshold
        threshold = 100 # adjust the threshold as needed
        if (closest_point[0]-x)**2 + (closest_point[1]-y)**2 < threshold:
            point_id = self.right_points.pop(closest_point, None)
            if point_id:
                self.canvas_right.delete(point_id)
                self.right_coords.remove(closest_point)
                self.update_table()
                self.display_images()
                
    def redraw_points(self, canvas, original_points, scaled_points):
        """Redraw points on the canvas after zooming"""
        canvas.delete("point")
        
        for (x, y), point_id in original_points.items():
            new_x = int(x * self.scale_factor)
            new_y = int(y * self.scale_factor)
            
            # Update the stored scaled points
            scaled_points[(x, y)] = (new_x, new_y)
            
            canvas.create_oval(
                new_x - 3, new_y - 3,
                new_x + 3, new_y + 3,
                fill=canvas.itemcget(point_id, "fill"),
                outline=canvas.itemcget(point_id, "outline"),
                tags="point"
            )

    def update_left_coord_label(self, event):
        """update left image coordinate label point from "left <Motion>" """
        # consider the scrollbar offset
        x = int((self.canvas_left.canvasx(event.x)) / self.scale_factor)
        y = int((self.canvas_left.canvasy(event.y)) / self.scale_factor)
        self.coord_label_left.config(text=f"Left Image Coordinates: ({x}, {y})")

        # draw the crosshair
        self.canvas_left.delete("crosshair")
        self.canvas_left.create_line(self.canvas_left.canvasx(event.x),                    # x0
                                     0,                                                    # y0
                                     self.canvas_left.canvasx(event.x),                    # x1
                                     self.canvas_left.winfo_height() * self.scale_factor,  # y1
                                     fill="red", tags="crosshair")
        self.canvas_left.create_line(0,                                                    # x0
                                     self.canvas_left.canvasy(event.y),                    # y0
                                     self.canvas_left.winfo_width() * self.scale_factor,   # x1
                                     self.canvas_left.canvasy(event.y),                    # y1
                                     fill="red", tags="crosshair")

    def update_right_coord_label(self, event):
        """update right image coordinate label point from "right <Motion>" """
        # consider the scrollbar offset
        x = int((self.canvas_right.canvasx(event.x)) / self.scale_factor)
        y = int((self.canvas_right.canvasy(event.y)) / self.scale_factor)
        self.coord_label_right.config(text=f"Right Image Coordinates: ({x}, {y})")

        # draw the crosshair
        self.canvas_right.delete("crosshair")
        self.canvas_right.create_line(self.canvas_right.canvasx(event.x),                   # x0
                                      0,                                                    # y0
                                      self.canvas_right.canvasx(event.x),                   # x1
                                      self.canvas_right.winfo_height() * self.scale_factor, # y1
                                      fill="red", tags="crosshair")
        self.canvas_right.create_line(0,                                                    # x0
                                      self.canvas_right.canvasy(event.y),                   # y0
                                      self.canvas_right.winfo_width() * self.scale_factor,  # x1
                                      self.canvas_right.canvasy(event.y),                   # y1
                                      fill="red", tags="crosshair")

    def update_table(self):
        """update the table point from "display_images", "mark_left", "mark_right", "delete_left_point", and "delete_right_point" """
        # write the coordinates to the table
        max_len = max(len(self.left_coords), len(self.right_coords))
        data = []
        
        for i in range(max_len):
            left_coord = self.left_coords[i] if i < len(self.left_coords) else ("", "")
            right_coord = self.right_coords[i] if i < len(self.right_coords) else ("", "")
            data.append([i+1, left_coord, right_coord])
            
        self.sheet.set_sheet_data(data)

    def export_to_csv(self):
        """export the coordinates to a CSV file point from "export_button" """
        file_path = filedialog.asksaveasfilename(initialfile = "transfer_file.csv", defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            # Prepare the data for export
            data = []
            for i in range(max(len(self.left_coords), len(self.right_coords))):
                left_coord = self.left_coords[i] if i < len(self.left_coords) else ("", "")
                right_coord = self.right_coords[i] if i < len(self.right_coords) else ("", "")
                data.append([i+1, left_coord, right_coord])

            # Create a pandas DataFrame
            df = pd.DataFrame(data, columns=["Point #", "Left Image (x, y)", "Right Image (x, y)"])

            # Export DataFrame to CSV
            df.to_csv(file_path, index=False)

    def zoom(self, factor):
        """zoom in/out the image point from "zoom_in_button" and "zoom_out_button" """
        self.scale_factor *= factor
        self.display_images()
        self.zoom_label.config(text=f"Zoom: {int(self.scale_factor * 100)}%")
        
    def start_drag_left(self, event):
        """Start dragging the left image when Ctrl is held and left mouse button is pressed point from "left <ButtonPress-1>" """
        if event.state & 0x0004:  # Check if Ctrl key is pressed
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
            self.is_dragging = True
        else:
            self.mark_left(event)

    def drag_left(self, event):
        """Drag the left image on the canvas point from "left <B1-Motion>" """
        if self.is_dragging:
            delta_x = (event.x - self.drag_data["x"])
            delta_y = (event.y - self.drag_data["y"])
            
            # Move the canvas view by the drag amount
            if (delta_x / self.scale_factor) > 0:
                self.canvas_left.xview_scroll(-1, "units")
            elif (delta_x / self.scale_factor) < 0:
                self.canvas_left.xview_scroll(1, "units")
            if (delta_y / self.scale_factor) > 0:
                self.canvas_left.yview_scroll(-1, "units")
            elif (delta_y / self.scale_factor) < 0:
                self.canvas_left.yview_scroll(1, "units")

            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y

    def end_drag_left(self, event):
        """End dragging the left image when the left mouse button is released point from "left <ButtonRelease-1>" """
        if self.is_dragging:
            self.is_dragging = False

    def start_drag_right(self, event):
        """Start dragging the right image when Ctrl is held and left mouse button is pressed point from "right <ButtonPress-1>" """
        if event.state & 0x0004:  # Check if Ctrl key is pressed
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
            self.is_dragging = True
        else:
            self.mark_right(event)

    def drag_right(self, event):
        """Drag the right image on the canvas point from "right <B1-Motion>" """
        if self.is_dragging:
            delta_x = event.x - self.drag_data["x"]
            delta_y = event.y - self.drag_data["y"]

            # Move the canvas view by the drag amount
            if (delta_x / self.scale_factor) > 0:
                self.canvas_right.xview_scroll(-1, "units")
            elif (delta_x / self.scale_factor) < 0:
                self.canvas_right.xview_scroll(1, "units")
            if (delta_y / self.scale_factor) > 0:
                self.canvas_right.yview_scroll(-1, "units")
            elif (delta_y / self.scale_factor) < 0:
                self.canvas_right.yview_scroll(1, "units")

            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y

    def end_drag_right(self, event):
        """End dragging the right image when the left mouse button is released point from "right <ButtonRelease-1>" """
        if self.is_dragging:
            self.is_dragging = False
            
    def load_csv_file(self):
        """Open a file dialog to load a CSV file and store its path in a variable."""
        file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
        
        if file_path:
            self.csv_file_path = file_path
            self.csv_file_label.config(text=f"Path: {file_path}")
            
    def image_process(self):
        before, after = self.method()
        fig, ax = plt.subplots(1, 3, figsize=(18, 7))
        ax[0].imshow(before)
        ax[1].imshow(after)
        ax[2].imshow(before, alpha=0.5)
        ax[2].imshow(after, alpha=0.5)
        
        # Create Toplevel window and store a reference to it
        plot_result = Toplevel(self.master)
        plot_result.title("Image Process Result")
        self.toplevel_windows.append(plot_result)
        
        canves1 = FigureCanvasTkAgg(fig, master=plot_result)
        canves1.draw()
        canves1.get_tk_widget().pack()
        
        # Create a Save Image button
        save_button = tk.Button(plot_result, text="Save Image", command=lambda: self.save_image(before, after))
        save_button.pack()
        
    def save_image(self, before, after):
        """Saves the figure as an image file."""
        file_path_1 = filedialog.asksaveasfilename(initialfile = "Before_warp.png", defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
        file_path_2 = filedialog.asksaveasfilename(initialfile = "After_warp.png", defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path_1 and file_path_2:
            # Save the figure using the selected file path
            image_save1 = Image.fromarray(before)
            image_save2 = Image.fromarray(after)
            image_save1.save(file_path_1)
            image_save2.save(file_path_2)
            print(f"Before Image saved to {file_path_1}")
            print(f"After Image saved to {file_path_2}")
    
    def check_process(self):
        before_image = np.array(self.original_left_img.copy())
        after_image = np.array(self.original_right_img.copy())
        self.method = process(before_image, after_image, self.csv_file_path, grid=(int(self.grid_cols), int(self.grid_rows)), fix_type=self.image_process_list.get())
        fig1 = self.method.check_plot()
        fig2 = self.method.check_plot2()
        
        # Create Toplevel windows and store references to them
        check_plot1 = Toplevel(self.master)
        check_plot2 = Toplevel(self.master)
        check_plot1.title("Check Image Process Region")
        check_plot2.title("Check Warp")
        self.toplevel_windows.append(check_plot1)
        self.toplevel_windows.append(check_plot2)
        
        canves1 = FigureCanvasTkAgg(fig1, master=check_plot1)
        canves2 = FigureCanvasTkAgg(fig2, master=check_plot2)
        canves1.draw()
        canves2.draw()
        canves1.get_tk_widget().pack()
        canves2.get_tk_widget().pack()
        
    def on_closing(self):
        """Handles the close event for the main window."""
        # Close all Toplevel windows
        for window in self.toplevel_windows:
            window.destroy()
        
        # Destroy the main window and exit the application
        self.master.destroy()
        sys.exit()
    
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLocatorApp(root)
    root.mainloop()