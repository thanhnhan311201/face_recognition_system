try:
    from src.system import face_recognition_system
except:
    from system import face_recognition_system

if __name__ == '__main__':
    dataset_path = 'datasets/'
    image_folder = 'datasets/images/'

    my_system = face_recognition_system(dataset_path, image_folder)

    my_system.index_dataset()