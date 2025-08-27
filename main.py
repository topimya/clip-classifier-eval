from tools import parse_args
from classify_and_eval import clip_main
from extract_objects import find_img_folder

print('Start')

args = parse_args()
orig_dir = args.path
save_clip_dir = args.path_res
FLAG = args.split
print('Вырезание объектов')
find_img_folder(args.path, save_clip_dir)
print('Классификациия CLIP')
clip_main(args.path, save_clip_dir)
print('End')