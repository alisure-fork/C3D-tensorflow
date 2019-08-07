import os
import random
from alisuretool.Tools import Tools

split_num = 4
data_dir = "/home/ubuntu/data1.5TB/C3D/UCF-101"

train_list, test_list = [], []
video_dir = [os.path.join(data_dir, _) for _ in sorted(os.listdir(data_dir))]
for video_index, video_one in enumerate(video_dir):
    video_one_dir = [os.path.join(video_one, _) for _ in sorted(os.listdir(video_one))]
    for video_one_dir_one in video_one_dir:
        if random.randint(0, split_num - 1) > 0:
            train_list.append("{} {}\n".format(video_one_dir_one, video_index))
        else:
            test_list.append("{} {}\n".format(video_one_dir_one, video_index))
            pass
        pass
    pass

Tools.write_to_txt("train.list", train_list, reset=True)
Tools.write_to_txt("test.list", test_list, reset=True)
