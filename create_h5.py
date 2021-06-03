import ThreeD60
import os
from torch.utils.data import DataLoader
import cv2
import shutil
import h5py
import numpy as np

DEPTH_PATH = "depth_path"
COLOR_PATH = "color_path"

# Type of picture positions we are using for generating our dataset. If you use more than one, batch_size must be 1
PLACEMENTS = [ThreeD60.Placements.CENTER, ThreeD60.Placements.RIGHT, ThreeD60.Placements.UP]

def create_numpy_array(output_file, input_files, sets, data_to_load, batch_size=1):
    with h5py.File(output_file, 'w') as hf:
        for i in range(len(input_files)):

            subgroup = hf.create_group(sets[i])

            names = []
            colors = []
            depths = []

            datasets = ThreeD60.get_datasets(input_files[i],
                                             datasets=data_to_load,
                                             placements=PLACEMENTS,
                                             image_types=[ThreeD60.ImageTypes.COLOR, ThreeD60.ImageTypes.DEPTH],
                                             longitudinal_rotation=False)

            print("Loaded %d samples." % len(datasets))

            dataset_loader = DataLoader(datasets, batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=False, num_workers=0)

            for it, b in enumerate(dataset_loader):

                for placement in PLACEMENTS:

                    color_paths = b[str(placement)][COLOR_PATH]

                    depth_batch_items = ThreeD60.extract_image(b, placement, ThreeD60.ImageTypes.DEPTH)

                    color_batch_items = ThreeD60.extract_image(b, placement,
                                                               ThreeD60.ImageTypes.COLOR)

                    for j in range(0, len(color_paths)):
                        names.append(color_paths[j])

                        cur_depth = depth_batch_items[j].cpu().numpy().transpose(1, 2, 0)
                        cur_depth = (255 - cur_depth)
                        depths.append(cur_depth)

                        cur_color = color_batch_items[j].cpu().numpy().transpose(1, 2, 0)
                        colors.append(cur_color)

            names = np.array(names, dtype=object)
            colors = np.array(colors)
            depths = np.array(depths)

            string_dt = h5py.string_dtype()
            subgroup.create_dataset("image_paths", data=names, dtype=string_dt)
            subgroup.create_dataset("colors", data=colors)
            subgroup.create_dataset("depths", data=depths)


if __name__ == "__main__":
    # need to change directories to get relative path names on the input files to work
    # this code assumes that the image and exr files are located one directory above
    os.chdir("..")
    # files containing the names of training, testing, and data splits
    # other files are located in ./3D60/splits/3dv19/
    # the difference between new and the original .txt is that one is that new uses / for unix
    # and the original files uses \ because it was designed for Windows machines.
    # please format the files similar to these files if you want to use this for other code
    input_files = [r"./3D60/splits/3dv19/new_train.txt",
                   r"./3D60/splits/3dv19/new_test.txt",
                   r"./3D60/splits/3dv19/new_val.txt"]
    # the sets directly correspond to the input files' order
    sets = ["train", "test", "val"]
    # name of the output file
    output_file = r"./3D60/matterport_dataset.h5"
    # the dataset we are using
    data_to_load = ["m3d"]
    # Sun, Matterport, #Standord are the only datasets we can use
    # data_to_load = ["suncg", "m3d", "s2d3d"]
    # batch size must be 1 if more than one placement in the constant PLACEMENTS array at the top of the file
    create_numpy_array(output_file, input_files, sets, data_to_load, batch_size=1)

    print("completed")
