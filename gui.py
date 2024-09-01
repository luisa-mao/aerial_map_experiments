import tkinter as tk
from PIL import Image, ImageTk
from simulator import Simulator

# Increase the maximum image size limit
Image.MAX_IMAGE_PIXELS = None


PATCH_SIZE = 32

root = tk.Tk()

# Variable to store whether the button has been clicked
draw_square = tk.BooleanVar()
draw_square.set(False)

set_start = tk.BooleanVar()
set_start.set(False)

set_goal = tk.BooleanVar()
set_goal.set(False)

# List to store the IDs of the drawn rectangles
rectangles = []
patch_centerpoints = []
image_patches = []

start = None
goal = None


def on_image_click(event):
    global draw_square
    global patch_centerpoints
    global rectangles
    global start
    global goal
    x = event.x
    y = event.y
    # print(f"Clicked at pixel coordinates: ({x}, {y})")

    # # print the x and y coordinates in the original image
    scale_factor = image.size[0] / resized_image.width()
    true_x = int(x * scale_factor)
    true_y = int(y * scale_factor)

    if draw_square.get():
        patch_centerpoints.append((true_x, true_y))

        # print(f"Clicked at image coordinates: ({true_x}, {true_y})")

        # print image width and resized image width
        print(f"Image width: {image.size[0]}")
        print(f"Resized image width: {resized_image.width()}")

        # print scale factor
        print(f"Scale factor: {scale_factor}")



        x_left = x - int(PATCH_SIZE / scale_factor / 2)
        x_right = x + int(PATCH_SIZE / scale_factor / 2)
        y_top = y - int(PATCH_SIZE / scale_factor / 2)
        y_bottom = y + int(PATCH_SIZE / scale_factor / 2)
        rectangle_id = canvas.create_rectangle(x_left, y_top, x_right, y_bottom, outline="blue")

        # print x right - x left
        print(f"Patch size: {x_right - x_left}")
        draw_square.set(False)

        rectangles.append(rectangle_id)  # Store the ID of the drawn rectangle
        display_patch()

    elif set_start.get():
        radius = 5  # Radius of the circle (dot)
        circle_id = canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="red")
        start = (true_x, true_y)
        print(f"Start: {start}")
        set_start.set(False)

    elif set_goal.get():
        radius = 5  # Radius of the circle (dot)
        circle_id = canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="red")
        goal = (true_x, true_y)
        print(f"Goal: {goal}")
        set_goal.set(False)

def on_button_click():
    global draw_square
    if len(rectangles) == 6:
        return
    draw_square.set(True)

def on_start_click():
    global set_start
    set_start.set(True)

def on_goal_click():
    global set_goal
    set_goal.set(True)

def display_patch():
    global current_square
    image_patch_center = patch_centerpoints[-1]
    x_left = image_patch_center[0] - PATCH_SIZE // 2
    x_right = image_patch_center[0] + PATCH_SIZE // 2
    y_top = image_patch_center[1] - PATCH_SIZE // 2
    y_bottom = image_patch_center[1] + PATCH_SIZE // 2

    # get the patch from the image
    patch = PIL_image.crop((x_left, y_top, x_right, y_bottom))
    # resize the patch to the size of the square
    # patch = patch.resize((PATCH_SIZE, PATCH_SIZE))
    # print the size of the patch
    print("patch size", patch.size)
    # Convert the patch to a format that tk.PhotoImage can handle
    patch = ImageTk.PhotoImage(patch)
    image_patches.append(patch)
    # display the patch in the current square
    squares[current_square].create_image(0, 0, image=patch, anchor='nw')
    current_square = len(rectangles)


def on_undo_click():
    global current_square
    if rectangles:  # If there are any drawn rectangles
        rectangle_id = rectangles.pop()  # Get the ID of the last drawn rectangle
        canvas.delete(rectangle_id)  # Delete the rectangle with this ID
        patch_centerpoints.pop()
        # make the current square black
        squares[current_square].create_rectangle(0, 0, PATCH_SIZE, PATCH_SIZE, fill="blue", outline="black", width=2)
        image_patches.pop()
        current_square = min(0, len(rectangles))

# Load the image
# image_path = "/scratch/luisamao/6-4-evening1/vis_images/22.png"
# image_path = "/scratch/luisamao/6-4-evening1/vis_images/13.png"
# image_path = "/scratch/luisamao/all_terrain/aerial_maps/map5.2.png"
image_path = "/robodata/ARL_SARA/2024/GQ/AeroPlan/r2c2/Stitched/A.png"
image = Image.open(image_path)
# print image size
# image = ImageTk.PhotoImage(image)
# print("Image size", image.width(), image.height())

# Load the image using PIL's Image class
PIL_image = Image.open(image_path)

# Create a canvas to display the image
canvas = tk.Canvas(root)
canvas.pack(fill=tk.BOTH, expand=True)

# Declare resized_image as a global variable
global resized_image


def on_run_simulator_click():

    global image_path
    global image

    # # print the x and y coordinates in the original image
    scale_factor = image.size[0] / resized_image.width()
    args = {
        "map_path": image_path,
        "mask_path": '../clean_mask.png',
        "scale": 1,
        "contrast": False,
        "log": False,
        "context": None
    }
    simulator = Simulator(None, args)
    # print(patch_centerpoints)
    path = simulator.run((start[1], start[0]), (goal[1], goal[0]), patch_centers=patch_centerpoints)

    # plot the path on the image
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        x1 = x1 / scale_factor
        y1 = y1 / scale_factor
        x2 = x2 / scale_factor
        y2 = y2 / scale_factor
        canvas.create_line(y1, x1, y2, x2, fill="red", width=2)


def resize_image(event):
    global resized_image  # Declare resized_image as global in the function
    global image
    # Calculate the desired width and height for the image
    canvas_width = event.width
    canvas_height = event.height
    # image_width = image.width()
    # image_height = image.height()
    image_width = image.size[0]
    image_height = image.size[1]

    # Calculate the scaling factor to fit the image within the canvas
    scale_factor = min(canvas_width / image_width, canvas_height / image_height)
    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)
    new_size = (new_width, new_height)

    

    # Resize the image
    resized_image = ImageTk.PhotoImage(image.resize(new_size))

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

# Create a "Set Start" button
start_button = tk.Button(root, text="Set Start", command=on_start_click)
start_button.pack()

# Create a "Set Goal" button
goal_button = tk.Button(root, text="Set Goal", command=on_goal_click)
goal_button.pack()

# Run simulator button
run_simulator_button = tk.Button(root, text="Run Simulator", command=on_run_simulator_click)
run_simulator_button.pack()

# Create a frame at the bottom of the root window
squares = []
current_square = 0
bottom_frame = tk.Frame(root)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Create a label above the grid
label = tk.Label(root, text="Preference Context")
label.pack(side=tk.BOTTOM)

# Create a 2x3 grid of black squares
for j in range(3):
    for i in range(2):
        square = tk.Canvas(bottom_frame, width=PATCH_SIZE, height=PATCH_SIZE)
        square.grid(row=i, column=j)
        square.create_rectangle(0, 0, PATCH_SIZE, PATCH_SIZE, fill="blue", outline="black", width=2)
        squares.append(square)

root.mainloop()