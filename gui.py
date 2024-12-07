from dataloader import get_dataloader
from estimator import SoilMoistureEstimator

import tkinter

from PIL import Image, ImageTk

def get_display_data(model):
    data_path = 'Data_i11_is/dataset_i11_is.csv'

    dataloader = get_dataloader(data_path)

    display_data = []
    for image, label in dataloader:
        predicted = model(image)

        image_pil = tensor_to_image(image[0])
        image_tk = ImageTk.PhotoImage(image_pil)
        display_data.append({'actual' : label, 'predicted' : predicted, 'image' : image_tk})

    return display_data

def tensor_to_image(tensor):
    tensor = tensor.clamp(0, 1) * 255
    array = tensor.numpy().astype('uint8')
    if array.ndim == 3:
        array = array.transpose(1, 2, 0)
    elif array.ndim == 4:
        array = array[0].transpose(1, 2, 0)
    return Image.fromarray(array)

def show_instance(data_frame, display_data, index):
    for widget in data_frame.winfo_children():
        widget.destroy()

    case = display_data[index]

    test_label = tkinter.Label(data_frame, text=f'Case {index}', font=('Arial', 16))
    test_label.pack()

    image_display = tkinter.Label(data_frame, image=case['image'])
    image_display.image = case['image']
    image_display.pack()

    actual_label = tkinter.Label(data_frame, text=f'Ground Truth: {case["actual"][0]}')
    actual_label.pack()

    predicted_label = tkinter.Label(data_frame, text=f'   Predicted: {case["predicted"]}')
    predicted_label.pack()

    navigation_frame = tkinter.Frame(data_frame)
    
    if index != 0:
        previous_button = tkinter.Button(navigation_frame, text='Previous', command=lambda: show_instance(data_frame, display_data, index - 1))
        previous_button.pack(side='left')

    if index != len(display_data) - 1:
        next_button = tkinter.Button(navigation_frame, text='Next', command=lambda: show_instance(data_frame, display_data, index + 1))
        next_button.pack(side='right')

    navigation_frame.pack()
    data_frame.pack()

def get_examiner_window(model=SoilMoistureEstimator.get_estimator()):
    window = tkinter.Tk()
    display_data = get_display_data(model)
    data_frame = tkinter.Frame(window)
    show_instance(data_frame, display_data, 0)
    return window