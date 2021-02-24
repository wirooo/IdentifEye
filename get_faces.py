from facenet_pytorch import MTCNN
import torch
import os.path


def create_mtcnn():
    """
    Creates and returns mtcnn model from facenet-pytorch
    :return: mtcnn model object
    """
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f"device: {device}")
    return MTCNN(keep_all=True, device=device)


def get_face(mtcnn, image, destination_dir, image_name, conf_thresh=0.9, size_thresh=0.05, image_size=(240, 240)):
    """
    Detects and saves faces from given image
    :param mtcnn: mtcnn model from facenet-pytorch
    :param image: PIL image object to detect from
    :param destination_dir: destination directory to save images to
    :param image_name: name of image file
    :param conf_thresh: solutions with confidence lower than conf_thresh are not saved
    :param size_thresh: solutions with sizes (by ratio) lower than size_thresh are not saved
    :param image_size: size of image to be saved
    :return:
    """
    bounding_boxes, conf = mtcnn.detect(image)
    if bounding_boxes is not None:
        for i in range(len(bounding_boxes)):
            cropped = image.crop(bounding_boxes[i])
            save_path = os.path.join(destination_dir, image_name.split(".")[-2] + f"_cropped_{i}.png")
            if (conf[i] > 0.99) or (conf[i] > conf_thresh and cropped.size[0]/image.size[0] > size_thresh and cropped.size[1]/image.size[1] > size_thresh):
                cropped.resize(image_size).save(save_path)
                print(f"Saving Image: {image_name}, face: {i}, conf:  {conf[i]}, size: {cropped.size}, size[0]: {cropped.size[0]/image.size[0]}, size[1]: {cropped.size[1]/image.size[1]}")
            else:
                print(f"Skipping: {image_name}, face: {i}, conf:  {conf[i]}, size: {cropped.size}, size[0]: {cropped.size[0]/image.size[0]}, size[1]: {cropped.size[1]/image.size[1]}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f"device: {device}")
    mtcnn = MTCNN(keep_all=True, device=device)
