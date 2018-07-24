from __future__ import print_function, division
import os
import sys
import subprocess

def video_process(dir_path, video_name):

  video_dir_path = os.path.join(dir_path, video_name)
  if not os.path.isdir(video_dir_path):
    return

  image_indices = []
  for image_file_name in os.listdir(video_dir_path):
    if 'n_frames' in image_file_name:
      continue
    image_indices.append(int(image_file_name[0:5]))

  if len(image_indices) == 0:
    print('no image files', video_dir_path)
    n_frames = 0
  else:
    image_indices.sort(reverse=True)
    n_frames = image_indices[0]
    print(video_dir_path, n_frames)
  with open(os.path.join(video_dir_path, 'n_frames'), 'w') as dst_file:
    dst_file.write(str(n_frames))


if __name__=="__main__":
  dir_path = sys.argv[1]
  for video_name in os.listdir(dir_path):
    video_process(dir_path, video_name)
