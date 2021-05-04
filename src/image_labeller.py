import PySimpleGUI as sg
import os
from shutil import copyfile
import argparse


class ImageLabeller:
    """
    A simple gui to label and categorize images of faces.
    """
    def __init__(self, img_dir, dest_dir, names):
        """
        Initializes gui window and formats file structure
        :param img_dir: path to raw images
        :param names: list containing names of people to label
        """

        photo_dir = os.listdir(img_dir)
        button_col = [
            [sg.Text(f"1/{len(photo_dir)}", size=(40, 1), key="-IMAGE_COUNT-")],
            [sg.Text("Identify the image as:")]
        ] + [[sg.Button(name, key=f"-NAME_{name}-")] for name in names] + [[sg.Button("Skip")]]

        image_viewer = [
            [sg.Text(photo_dir[0], size=(40, 1), key="-IMAGE_NAME-")],
            [sg.Image(key="-IMAGE-")]
        ]
        layout = [
            [
                sg.Column(image_viewer),
                sg.VSeparator(),
                sg.Column(button_col)
            ]
        ]
        self.window = sg.Window("Image Labeller", layout)
        self.img_dir = img_dir
        self.dest_dir = dest_dir
        self.names = names

        # cleanup empty folders
        for dir in os.listdir(dest_dir):
            pth = os.path.join(dest_dir, dir)
            if os.path.isdir(pth) and not os.listdir(pth):
                os.rmdir(pth)
        # create a folder for each name
        for name in names:
            pth = os.path.join(dest_dir, name)
            if not os.path.exists(pth):
                os.mkdir(pth)

    def show(self):
        """
        Runs the gui and processes user input
        """
        dir = os.listdir(self.img_dir)
        photo_dir = []
        for item in dir:
            if item != ".gitignore":
                print(item)
                photo_dir.append(item)

        i = 0
        self.window.read()
        while i < len(photo_dir):
            image_file_name = photo_dir[i]
            self.window["-IMAGE_COUNT-"].update(f"{i+1}/{len(os.listdir(self.img_dir))}")
            self.window["-IMAGE_NAME-"].update(image_file_name)
            self.window["-IMAGE-"].update(filename=os.path.join(self.img_dir, image_file_name))
            event, values = self.window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            elif event.split("_")[0] == "-NAME":
                name = event.split("_")[1][:-1]
                copyfile(os.path.join(self.img_dir, image_file_name), os.path.join(self.dest_dir, name, image_file_name))
                i += 1
            elif event == "Skip":
                i += 1
        self.window.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo_dir", type=str, nargs="?", default="../photos/faces")
    parser.add_argument("--label_dir", type=str, nargs="?", default="../photos/labelled")
    parser.add_argument("names", type=str, nargs="+")
    args = parser.parse_args()
    gui = ImageLabeller(args.photo_dir, args.label_dir, args.names)
    gui.show()
