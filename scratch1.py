import tkinter as tk
from PIL import Image, ImageTk
from simulator import Simulator
import cv2

# Increase the maximum image size limit
Image.MAX_IMAGE_PIXELS = None


PATCH_SIZE = 64

root = tk.Tk()

draw_square = tk.BooleanVar()


# List to store the IDs of the drawn rectangles
rectangles = []
patch_centerpoints = []
image_patches = []


def on_image_click(event):
    if not draw_square.get():
        return

    global patch_centerpoints
    global rectangles
    x = event.x
    y = event.y
    # print(f"Clicked at pixel coordinates: ({x}, {y})")

    # # print the x and y coordinates in the original image
    scale_factor = image.width() / resized_image.width()
    true_x = int(x * scale_factor)
    true_y = int(y * scale_factor)

    patch_centerpoints.append((true_x, true_y))

    print(len(patch_centerpoints), f"Clicked at image coordinates: ({true_x}, {true_y})")

    # # print image width and resized image width
    # print(f"Image width: {image.width()}")
    # print(f"Resized image width: {resized_image.width()}")

    # # print scale factor
    # print(f"Scale factor: {scale_factor}")



    x_left = x - int(PATCH_SIZE / scale_factor / 2)
    x_right = x + int(PATCH_SIZE / scale_factor / 2)
    y_top = y - int(PATCH_SIZE / scale_factor / 2)
    y_bottom = y + int(PATCH_SIZE / scale_factor / 2)
    rectangle_id = canvas.create_rectangle(x_left, y_top, x_right, y_bottom, outline="blue")

    # print x right - x left
    # print(f"Patch size: {x_right - x_left}")

    rectangles.append(rectangle_id)  # Store the ID of the drawn rectangle


def on_button_click():
    draw_square.set(True)

def on_save_click():
    global image_path
    global patch_centerpoints
    # load the map using cv2
    map = cv2.imread(image_path)
    # bgr to rgb
    map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
    map = map / 255.0
    for i, centerpoint in enumerate(patch_centerpoints):
        x, y = centerpoint
        x_left = x - int(PATCH_SIZE / 2)
        x_right = x + int(PATCH_SIZE / 2)
        y_top = y - int(PATCH_SIZE / 2)
        y_bottom = y + int(PATCH_SIZE / 2)
        patch = map[y_top:y_bottom, x_left:x_right]
        image_patches.append((patch, i//10))

    print("saved image patches", len(image_patches))

    # pickle the image patches
    import pickle
    import os
    save_dir = "/scratch/luisamao/all_terrain/sterling_unseen_dataset/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # count the number of files in the directory
    num_files = len([name for name in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, name))])
    with open(save_dir + f"image_patches_{num_files}.pkl", "wb") as f:
        pickle.dump(image_patches, f)
                    




def on_undo_click():
    if rectangles:  # If there are any drawn rectangles
        rectangle_id = rectangles.pop()  # Get the ID of the last drawn rectangle
        canvas.delete(rectangle_id)  # Delete the rectangle with this ID
        patch_centerpoints.pop()

# Load the image
image_path = "/scratch/luisamao/all_terrain/aerial_maps/map1.1.png"
# image_path = "/scratch/luisamao/all_terrain/aerial_maps/EER2.png"
image = tk.PhotoImage(file=image_path)

# Load the image using PIL's Image class
PIL_image = Image.open(image_path)

# Create a canvas to display the image
canvas = tk.Canvas(root)
canvas.pack(fill=tk.BOTH, expand=True)

# Declare resized_image as a global variable
global resized_image


def resize_image(event):
    global resized_image  # Declare resized_image as global in the function
    # Calculate the desired width and height for the image
    canvas_width = event.width
    canvas_height = event.height
    image_width = image.width()
    image_height = image.height()

    # Calculate the scaling factor to fit the image within the canvas
    scale_factor = min(canvas_width / image_width, canvas_height / image_height)
    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)

    

    # Resize the image
    resized_image = image.subsample(int(1/scale_factor))

    # Display the image on the canvas
    canvas.delete("all")  # Clear the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=resized_image)

# Bind the resize event to the canvas
canvas.bind("<Configure>", resize_image)

# Bind the click event to the canvas
canvas.bind("<Button-1>", on_image_click)

# Create a button that enables drawing a square on the image
button = tk.Button(root, text="Choose Patch", command=on_button_click)
button.pack()

# Create an "Undo" button
undo_button = tk.Button(root, text="Undo", command=on_undo_click)
undo_button.pack()

# Create an "Save" button
save_button = tk.Button(root, text="Save", command=on_save_click)
save_button.pack()


root.mainloop()