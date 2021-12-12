# Portfolio
Проект "Удаление автомобиля из видео"

GAN_7. 256 х 384.

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)
