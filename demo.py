import cv2

try:
    from src.system import face_recognition_system
except:
    from system import face_recognition_system

if __name__ == '__main__':
    dataset_path = 'datasets/'
    image_folder = 'datasets/images/'

    my_system = face_recognition_system(dataset_path, image_folder)

    demo_img = 'demo_imgs/demo2.jpg'
    my_system.recognize_face_via_image(demo_img)

    # my_system.recognize_face_via_camera()