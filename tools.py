import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Save cut img with text and save metricks")
    parser.add_argument("path", type=str, help="Путь до папки с изображениями")
    parser.add_argument("path_res", type=str, help="Путь до папки, где сохранить результат CLIP (метрики и папки с изображением)")
    parser.add_argument("--split", action="store_true", help="Разделяет класс на два и более, если они через ','. По умолчанию не разедяет. Чтобы разедилть (True)")
    
    args = parser.parse_args()

    return args