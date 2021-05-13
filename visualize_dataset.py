# import ThreeD60
# import visualization
# import os
# from torch.utils.data import DataLoader
# import cv2
# import shutil
# import h5py
# import numpy as np
#
# PLACEMENT = "left_down"
# DEPTH_PATH = "depth_path"
# COLOR_PATH = "color_path"
# DEPTH_CONST = 50

import ThreeD60
import visualization

from torch.utils.data import DataLoader

if __name__ == "__main__":
    datasets = ThreeD60.get_datasets(".//splits//eccv18//caffe_train_allmods_abs.txt", \
                                     datasets=["suncg", "m3d", "s2d3d"],
                                     placements=[ThreeD60.Placements.CENTER, ThreeD60.Placements.RIGHT,
                                                 ThreeD60.Placements.UP],
                                     image_types=[ThreeD60.ImageTypes.COLOR, ThreeD60.ImageTypes.DEPTH,
                                                  ThreeD60.ImageTypes.NORMAL], longitudinal_rotation=False)
    print("Loaded %d samples." % len(datasets))

    viz = visualization.VisdomImageVisualizer("3D60", "127.0.0.1")
    dataset_loader = DataLoader(datasets, batch_size=32, shuffle=True, pin_memory=False, num_workers=4)
    for i, b in enumerate(dataset_loader):
        viz.show_images_grid(ThreeD60.extract_image(b, ThreeD60.Placements.CENTER, ThreeD60.ImageTypes.COLOR), "all_colors")
        viz.show_depths_grid(ThreeD60.extract_image(b, ThreeD60.Placements.CENTER, ThreeD60.ImageTypes.DEPTH), "all_depths")
        viz.show_normals_grid(ThreeD60.extract_image(b, ThreeD60.Placements.CENTER, ThreeD60.ImageTypes.NORMAL), "all_normals")
#
# #
# # def create_img_depth_maps(input_file, output_dir, data_to_load, batch_size=64):
# #     datasets = ThreeD60.get_datasets(input_file,
# #                                      datasets=data_to_load,
# #                                      placements=[ThreeD60.Placements.CENTER],
# #                                      # placements=[ThreeD60.Placements.CENTER, ThreeD60.Placements.RIGHT, ThreeD60.Placements.UP]
# #                                      image_types=[ThreeD60.ImageTypes.COLOR,
# #                                                   ThreeD60.ImageTypes.DEPTH],
# #                                      longitudinal_rotation=False)
# #
# #     print("Loaded %d samples." % len(datasets))
# #     dataset_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True,
# #                                 pin_memory=False, num_workers=0)
# #
# #     for it, b in enumerate(dataset_loader):
# #
# #         depth_paths = b[PLACEMENT][DEPTH_PATH]
# #         color_paths = b[PLACEMENT][COLOR_PATH]
# #         depth_batch_items = ThreeD60.extract_image(b,
# #                                                    ThreeD60.Placements.CENTER,
# #                                                    ThreeD60.ImageTypes.DEPTH)
# #
# #         color_batch_items = ThreeD60.extract_image(b,
# #                                                    ThreeD60.Placements.CENTER,
# #                                                    ThreeD60.ImageTypes.COLOR)
# #
# #         for i in range(0, len(depth_paths)):
# #
# #             cur_item = color_batch_items[i].cpu().numpy().transpose(1, 2, 0)
# #             print(cur_item.shape)
# #             cur_item = cv2.cvtColor(cur_item, cv2.COLOR_RGB2BGR)
# #             cv2.imshow(color_paths[i], cur_item)
# #             cv2.waitKey()
# #
# #         # print("#########")
# #         # base = os.path.splitext(os.path.basename(depth_paths[i]))[0]
# #         # cur_img_base = os.path.splitext(depth_paths[i])[0]
# #         #
# #         # cur_img_path = cur_img_base.replace("_depth_", "_color_") + ".png" # old color png img
# #         # new_img_path = os.path.join(output_dir, os.path.basename(cur_img_path)) # copy color png
# #         # new_depth_path = os.path.join(output_dir,  base + ".png") #depth png name
# #         # print(cur_img_path, new_img_path, new_depth_path)
# #         #
# #         # cur_item = depth_batch_items[i].cpu().numpy().transpose(1, 2, 0)
# #         # print(cur_item)
# #         # # cur_item = (255 - (cur_item *DEPTH_CONST))
# #         # cur_item = (255 - cur_item)
# #         # print(cur_item)
# #         # cv2.imwrite(new_depth_path, cur_item)
# #         # print(os.getcwd())
# #         # shutil.copyfile(cur_img_path, new_img_path)
# #         # # shutil.move(cur_img_path, new_img_path)
#
#
# #
# #
# # # # data2load array of arrays  #array of nums
# # # #set train, test, val
# # # color is rgb scaled 0 to 1 like opencv, if relying on opencv convert to bgr
# # # depth is 255 - num_loaded (white is closer) by hvr file, some extremely large numbers in there 1to e10 magniutude, but img looks normal enough
#
# def create_numpy_array(output_file, input_files, sets, data_to_load,  batch_size=64):
#
#     with h5py.File(output_file, 'w') as hf:
#         for i in range(len(input_files)):
#
#             subgroup = hf.create_group(sets[i])
#
#             names = []
#             colors = []
#             depths = []
#
#             datasets = ThreeD60.get_datasets(input_files[i],
#                                              datasets=data_to_load,
#                                              placements=[ThreeD60.Placements.CENTER],
#                                              # placements=[ThreeD60.Placements.CENTER, ThreeD60.Placements.RIGHT, ThreeD60.Placements.UP]
#                                              image_types=[ThreeD60.ImageTypes.COLOR, ThreeD60.ImageTypes.DEPTH],
#                                              longitudinal_rotation=False)
#
#             print("Loaded %d samples." % len(datasets))
#
#             dataset_loader = DataLoader(datasets, batch_size=1,
#                                         shuffle=False,
#                                         pin_memory=False, num_workers=0)
#             one = 0
#             for it, b in enumerate(dataset_loader):
#                 # if it is "str" or b is None:
#                 #     continue
#                 one += 1
#                 depth_paths = b[PLACEMENT][DEPTH_PATH]
#                 color_paths = b[PLACEMENT][COLOR_PATH]
#                 depth_batch_items = ThreeD60.extract_image(b, ThreeD60.Placements.CENTER,
#                                                            ThreeD60.ImageTypes.DEPTH)
#
#                 color_batch_items = ThreeD60.extract_image(b, ThreeD60.Placements.CENTER,
#                                                            ThreeD60.ImageTypes.COLOR)
#
#                 for j in range(0, len(depth_paths)):
#                     print(one,j)
#                     names.append(color_paths[j])  # .\dataset_dir\img_structure to png
#
#                     cur_depth = depth_batch_items[j].cpu().numpy().transpose(1, 2, 0)
#                     # cur_depth= cur_depth *DEPTH_CONST))
#                     cur_depth = (255 - cur_depth)
#                     depths.append(cur_depth)
#
#                     cur_color = color_batch_items[j].cpu().numpy().transpose(1, 2, 0)
#                     colors.append(cur_color)
#
#             names = np.array(names, dtype=object)
#             colors = np.array(colors)
#             depths = np.array(depths)
#
#             string_dt = h5py.string_dtype()
#             subgroup.create_dataset("image_paths", data=names, dtype=string_dt)
#             subgroup.create_dataset("colors", data=colors)
#             subgroup.create_dataset("depths", data=depths)
#
#
#
#
# if __name__ == "__main__":
#
#     os.chdir("..")
#
#     input_files = [r".\3D60\splits\3dv19\train.txt",
#                    r".\3D60\splits\3dv19\test_copy.txt",
#                    r".\3D60\splits\3dv19\val.txt"]
#
#     sets = ["train", "test", "val"]
#     output_file = r".\3D60\dataset_no_suncg.h5"
#
#     data_to_load = ["m3d", "s2d3d"]
#     # data_to_load = ["suncg", "m3d", "s2d3d"]
#
#     create_numpy_array(output_file, input_files, sets, data_to_load, batch_size=64)
#
#     print("commpleted")
#     # # datasets = ThreeD60.get_datasets(r".\3D60\splits\3dv19\test.txt",
#     # #     datasets=["s2d3d"], #datasets=["suncg", "m3d", "s2d3d"],
#     # #     placements=[ThreeD60.Placements.CENTER], # placements=[ThreeD60.Placements.CENTER, ThreeD60.Placements.RIGHT, ThreeD60.Placements.UP]
#     # #     image_types=[ThreeD60.ImageTypes.COLOR, ThreeD60.ImageTypes.DEPTH, ThreeD60.ImageTypes.NORMAL], longitudinal_rotation=False)
#     # # print("Loaded %d samples." % len(datasets))
#     #
#     # # #viz = visualization.VisdomImageVisualizer("3D60")
#     # # dataset_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True,
#     # #                             pin_memory=False, num_workers=0)
#     # #
#     # # for i, b in enumerate(dataset_loader): # each b is set of 32 imgs
#     # #     # viz.show_images_grid(ThreeD60.extract_image(b, ThreeD60.Placements.CENTER, ThreeD60.ImageTypes.COLOR), "all_colors")
#     # #     # viz.show_depths_grid(ThreeD60.extract_image(b, ThreeD60.Placements.CENTER, ThreeD60.ImageTypes.DEPTH), "all_depths")
#     # #     # viz.show_normals_grid(ThreeD60.extract_image(b, ThreeD60.Placements.CENTER, ThreeD60.ImageTypes.NORMAL), "all_normals")
#     # #
#     # #     #remake color img if needed
#     # #     print(b["left_down"]["color_path"][0])
#     # #     item = ThreeD60.extract_image(b, ThreeD60.Placements.CENTER, ThreeD60.ImageTypes.NORMAL)
#     # #     item_0 = item[0].cpu().numpy().transpose(1, 2, 0) * 255
#     # #     cv2.imwrite(r".\3D60\windowNORMAL_A.png",item_0)
#     # #     cv2.imwrite(r".\3D60\windowNORMAL_B.png",cv2.cvtColor(item_0, cv2.COLOR_BGR2RGB))
#     # #     depth_const = 50
#     # #     #remake depth img if needed
#     # #     print(b["left_down"]["depth_path"][0])
#     # #     item = ThreeD60.extract_image(b, ThreeD60.Placements.CENTER, ThreeD60.ImageTypes.DEPTH)
#     # #     item_0 = item[0].cpu().numpy().transpose(1, 2, 0) * depth_const
#     # #     cv2.imwrite(r".\3D60\window0.png", item_0/2)
#     # #     cv2.imwrite(r".\3D60\window1.png",item_0)
#     # #     cv2.imwrite(r".\3D60\window2.png", item_0*2)
#     # #
#     # #     item_0 = (255 - item_0)
#     # #     cv2.imwrite(r".\3D60\window-0.png", item_0/2)
#     # #     cv2.imwrite(r".\3D60\window-1.png",item_0)
#     # #     cv2.imwrite(r".\3D60\window-2.png", item_0*2)
#     # #     break
