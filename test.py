import os
import cv2
import dlib
import uuid
import torch
import numpy as np
from time import time
from torchvision import datasets
from torch.autograd import Variable
from skimage import io, img_as_ubyte
from skimage.transform import resize
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, ToPILImage


def load_data(batch_size=32, source='data/celeb'):
    data = datasets.ImageFolder(root=source,
                                transform=Compose([Resize((224, 224)),
                                                   RandomHorizontalFlip(),
                                                   ToTensor()]))

    data_loader = DataLoader(data,
                             batch_size=batch_size,
                             shuffle=False)

    return data_loader


def visualize(data_loader=None):
    images, label = iter(data_loader).next()
    idx = 0
    transform = ToPILImage()
    img = transform(images[idx])
    img.show()


def test_batch(root='', src='', dst='', model=None, tags=[], device='cuda'):
    transform = Compose([Resize((224, 224)),
                         RandomHorizontalFlip(),
                         ToTensor()])
    back2image = ToPILImage()
    face_detector = dlib.get_frontal_face_detector()

    for image in os.listdir(src):
        img_path = os.path.join(src, image)
        img = io.imread(img_path)

        scale_percent_dec = 50  # percent of original size
        width_dec = int(img.shape[1] * scale_percent_dec / 100)
        height_dec = int(img.shape[0] * scale_percent_dec / 100)
        dim_dec = (width_dec, height_dec)

        scale_percent_inc = 200  # percent of original size
        width_inc = int(img.shape[1] * scale_percent_inc / 100)
        height_inc = int(img.shape[0] * scale_percent_inc / 100)
        dim_inc = (width_inc, height_inc)

        # resize image
        if img.shape[1] >= 1920 and img.shape[0] >= 1080:
            img = cv2.resize(img, dim_dec, interpolation=cv2.INTER_AREA)
        elif img.shape[1] <= 640 and img.shape[0] <= 360:
            img = cv2.resize(img, dim_inc, interpolation=cv2.INTER_AREA)

        img_copy = img.copy()

        faces = face_detector(img, 0)

        if len(faces) == 1:
            print("Found {} face".format(len(faces)))
        else:
            print("Found {} faces".format(len(faces)))

        for face in faces:
            top, bottom, left, right = face.top(), face.bottom(), face.left(), face.right()
            face_image = img[top:bottom, left:right]
            face_image = resize(face_image, (299, 299), anti_aliasing=True, mode='constant')
            face_image = img_as_ubyte(face_image)
            pil_img = back2image(face_image)
            pil_img = transform(pil_img)
            pil_img = pil_img.view(1, 3, 224, 224)
            pil_img = Variable(pil_img.to(device))
            output = model(pil_img)
            pred_prob, pred_label = torch.max(output, dim=1)
            print(tags[pred_label[0]])
            cv2.rectangle(img_copy, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(img_copy, tags[pred_label[0]], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(os.path.join(dst, tags[pred_label[0]] + '.jpg'), face_image)
            cv2.imwrite(os.path.join(root, str(uuid.uuid4()) + '.jpg'), face_image)
        image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(root, str(uuid.uuid4()) + '.jpg'), image)


def test_one(root='', dst='', source='', model=None, tags=[], device='cuda'):
    img = io.imread(source)
    transform = Compose([Resize((224, 224)),
                         RandomHorizontalFlip(),
                         ToTensor()])
    back2image = ToPILImage()

    face_detector = dlib.get_frontal_face_detector()

    scale_percent_dec = 50  # percent of original size
    width_dec = int(img.shape[1] * scale_percent_dec / 100)
    height_dec = int(img.shape[0] * scale_percent_dec / 100)
    dim_dec = (width_dec, height_dec)

    scale_percent_inc = 200  # percent of original size
    width_inc = int(img.shape[1] * scale_percent_inc / 100)
    height_inc = int(img.shape[0] * scale_percent_inc / 100)
    dim_inc = (width_inc, height_inc)

    # resize image
    if img.shape[1] >= 1920 and img.shape[0] >= 1080:
        img = cv2.resize(img, dim_dec, interpolation=cv2.INTER_AREA)
    elif img.shape[1] <= 640 and img.shape[0] <= 360:
        img = cv2.resize(img, dim_inc, interpolation=cv2.INTER_AREA)

    img_copy = img.copy()

    alpha, beta = 2, 0
    # img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # blur = cv2.GaussianBlur(img, (19, 19), 0)
    faces = face_detector(img, 0)

    if len(faces) == 1:
        print("Found {} face".format(len(faces)))
    else:
        print("Found {} faces".format(len(faces)))

    for face in faces:
        top, bottom, left, right = face.top(), face.bottom(), face.left(), face.right()
        face_image = img[top:bottom, left:right]
        face_image = resize(face_image, (299, 299), anti_aliasing=True, mode='constant')
        face_image = img_as_ubyte(face_image)
        pil_img = back2image(face_image)
        pil_img = transform(pil_img)
        pil_img = pil_img.view(1, 3, 224, 224)
        pil_img = Variable(pil_img.to(device))
        output = model(pil_img)
        pred_prob, pred_label = torch.max(output, dim=1)
        print(tags[pred_label[0]])
        cv2.rectangle(img_copy, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(img_copy, tags[pred_label[0]], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dst, tags[pred_label[0]] + '.jpg'), face_image)
    image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(root, str(uuid.uuid4()) + '.jpg'), image)


def test_video(model=None, tags=[], src='', root='', dst='', device='cuda'):
    transform = Compose([Resize((224, 224)),
                         RandomHorizontalFlip(),
                         ToTensor()])
    back2image = ToPILImage()

    face_detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(src)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    c, num_faces = 0, 0
    start = time()
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            img = frame

            scale_percent_dec = 20  # percent of original size
            width_dec = int(img.shape[1] * scale_percent_dec / 100)
            height_dec = int(img.shape[0] * scale_percent_dec / 100)
            dim_dec = (width_dec, height_dec)

            scale_percent_inc = 200  # percent of original size
            width_inc = int(img.shape[1] * scale_percent_inc / 100)
            height_inc = int(img.shape[0] * scale_percent_inc / 100)
            dim_inc = (width_inc, height_inc)

            # resize image
            if img.shape[1] >= 1920 and img.shape[0] >= 1080:
                img = cv2.resize(img, dim_dec, interpolation=cv2.INTER_AREA)
            elif img.shape[1] <= 640 and img.shape[0] <= 360:
                img = cv2.resize(img, dim_inc, interpolation=cv2.INTER_AREA)

            img_copy = img.copy()

            alpha, beta = 2, 1
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

            img = cv2.GaussianBlur(img, (11, 11), 0)
            faces = face_detector(img, 0)

            if len(faces) == 1:
                print("Found {} face".format(len(faces)))
            else:
                print("Found {} faces".format(len(faces)))

            for face in faces:
                top, bottom, left, right = face.top(), face.bottom(), face.left(), face.right()
                face_image = img[top:bottom, left:right]
                face_image = resize(face_image, (299, 299), anti_aliasing=True, mode='constant')
                face_image = img_as_ubyte(face_image)
                pil_img = back2image(face_image)
                pil_img = transform(pil_img)
                pil_img = pil_img.view(1, 3, 224, 224)
                pil_img = Variable(pil_img.to(device))
                output = model(pil_img)

                pred_prob, pred_label = torch.max(output, dim=1)
                print(tags[pred_label[0]])
                cv2.rectangle(img_copy, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(img_copy, tags[pred_label[0]], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                # cv2.imwrite(os.path.join(dst, str(uuid.uuid4()) + '.jpg'), face_image)

            cv2.imwrite(os.path.join(root, str(c) + '.jpg'), img_copy)
            cv2.imwrite(os.path.join(root, str(c+100) + '.jpg'), img)
            # cv2.imshow('Frame', img)
            c += 1
            num_faces += len(faces)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    end = time()
    print('len = {}, time = {}, fps = {}'.format(num_faces, end - start, num_faces / (end - start)))


def frames_to_video(src=''):
    frame_list = []
    fps = 24
    output_path = 'data/test/video.mp4'
    size = 0
    start = time()
    print('Creating video from frames....')
    for frame in sorted(os.listdir(src), key=len):
        img = cv2.imread(os.path.join(src, frame))
        height, width, layers = img.shape
        size = (width, height)
        frame_list.append(img)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for images in frame_list:
        out.write(images)
    out.release()
    end = time()
    print('{} seconds'.format(end - start))


def main():
    device = 'cuda'
    root = 'data/test'
    frames_root = 'data/test/frames_exp'
    src = 'data/test_data/images'
    dst = 'data/test/data'
    test_image_path = 'data/test_data/images/bvs.jpg'
    test_video_path = 'data/test_data/videos/bvs1.mp4'
    tags = np.load('data/tags.npy')
    model = torch.load('model/model.pt').to(device)
    model.eval()
    # test_batch(root=root, src=src, dst=dst, model=model, tags=tags, device=device)
    # test_one(root=root, dst=dst, source=test_image_path, model=model, tags=tags, device=device)
    # test_video(model=model, tags=tags, root=frames_root, src=test_video_path, dst=dst, device=device)
    frames_to_video(src=frames_root)


if __name__ == '__main__':
    print('Testing on {}'.format(torch.cuda.get_device_name(0)))
    main()
