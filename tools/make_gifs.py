import argparse
import cv2
import imageio.v2 as imageio
import os

def create_mp4(input_folder, prefix, output_mp4, fps=60):
    """ 生成 60 FPS 的 MP4，约 2 秒播放完 """
    files = [f for f in os.listdir(input_folder) 
             if f.endswith('.tga') and f.startswith(prefix)]
    files.sort(key=lambda x: int(x[len(prefix):-4]))  # 按数字排序

    # 读取第一张图片获取尺寸
    first_img = cv2.imread(os.path.join(input_folder, files[0]))
    height, width = 960, 720

    # 创建 VideoWriter（H.264 编码）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'avc1'（兼容性更好）
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))

    for filename in files:
        img = cv2.imread(os.path.join(input_folder, filename))
        out.write(img)
    
    out.release()
    print(f"MP4 已生成: {output_mp4} | 总帧数: {len(files)} | FPS: {fps} | 时长: {len(files)/fps:.2f} 秒")

def create_gif_from_prefix(input_folder, prefix, output_filename, fps=60):
    """
    使用imageio创建更流畅的GIF
    
    参数:
        fps: 帧率(帧/秒)，默认30fps
    """
    files = [f for f in os.listdir(input_folder) 
            if f.endswith('.tga') and f.startswith(prefix)]
    files.sort(key=lambda x: int(x[len(prefix):-4]))
    
    if not files:
        print(f"未找到以'{prefix}'开头的TGA文件")
        return
    
    images = []
    for filename in files:
        images.append(imageio.imread(os.path.join(input_folder, filename)))
    
    # 计算每帧持续时间(秒)
    duration = 1 / fps
    
    # 保存GIF
    imageio.mimsave(output_filename, images, duration=duration)
    print(f"GIF 已生成: {output_filename} | 总帧数: {len(files)} | FPS: {fps} | 时长: {len(files)/fps:.2f} 秒")

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', default='')
args = parser.parse_args()


# 创建gt_系列的GIF
create_gif_from_prefix(args.src_dir, "gt_", args.src_dir + "gt_animation.gif", fps=60)

# 创建pred_系列的GIF
create_gif_from_prefix(args.src_dir, "pred_", args.src_dir + "pred_animation.gif", fps=60)

# create_mp4(args.src_dir, "gt_", args.src_dir + "gt_animation.mp4", fps=60)
# create_mp4(args.src_dir, "pred_", args.src_dir + "pred_animation.mp4", fps=60)
