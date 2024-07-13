import os

def save_image(image, save_path, index):
    image_name = str(index) + '.png'
    store_path = os.path.join(save_path, image_name)
    image.save(store_path)


def is_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created.")
    else:
        print(f"Directory '{folder_path}' already exists.")