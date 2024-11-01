import os, sys, argparse
import random
import csv
import numpy as np
from scipy.stats import bernoulli
from helper_functions import find_files
from extract_leads import get_paper_ecg
from HandwrittenText.generate import get_handwritten
from CreasesWrinkles.creases import get_creased
from ImageAugmentation.augment import get_augment
import warnings
import cv2
import random

from PIL import Image
from openpyxl import load_workbook

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-hea', '--header_file', type=str, required=True)
    parser.add_argument('-o', '--output_directory', type=str, required=True)
    parser.add_argument('-se', '--seed', type=int, required=False, default=-1)
    parser.add_argument('-st', '--start_index', type=int, required=True, default=-1)
    parser.add_argument('--num_leads', type=str, default='twelve')

    parser.add_argument('-r', '--resolution', type=int, required=False, default=200)
    parser.add_argument('--pad_inches', type=int, required=False, default=0)
    parser.add_argument('-ph', '--print_header', action="store_true", default=False)
    parser.add_argument('--num_columns', type=int, default=-1)
    parser.add_argument('--full_mode', type=str, default='II')

    parser.add_argument('-l', '--link', type=str, required=False,
                        default='https://www.physionet.org/content/ptbdb/1.0.0/')
    parser.add_argument('-n', '--num_words', type=int, required=False, default=5)
    parser.add_argument('--x_offset', dest='x_offset', type=int, default=30)
    parser.add_argument('--y_offset', dest='y_offset', type=int, default=30)
    parser.add_argument('--hws', dest='handwriting_size_factor', type=float, default=0.2)

    parser.add_argument('-ca', '--crease_angle', type=int, default=90)
    parser.add_argument('-nv', '--num_creases_vertically', type=int, default=10)
    parser.add_argument('-nh', '--num_creases_horizontally', type=int, default=10)

    parser.add_argument('-rot', '--rotate', type=int, default=0)
    parser.add_argument('-noise', '--noise', type=int, default=50)
    parser.add_argument('-c', '--crop', type=float, default=0.01)
    parser.add_argument('-t', '--temperature', type=int, default=40000)

    parser.add_argument('--random_resolution', action="store_true", default=False)
    parser.add_argument('--random_padding', action="store_true", default=False)
    parser.add_argument('--random_grid_color', action="store_true", default=False)
    parser.add_argument('--standard_grid_color', type=int, default=5)
    parser.add_argument('--random_dc', type=float, default=0)
    parser.add_argument('--random_grid_present', type=float, default=1)
    parser.add_argument('--random_add_header', type=float, default=0)
    parser.add_argument('--random_bw', type=float, default=0)
    parser.add_argument('--deterministic_lead', action="store_false", default=True)
    parser.add_argument('--store_text_bounding_box', action="store_true", default=False)
    parser.add_argument('--store_config', action="store_true", default=False)

    parser.add_argument('--deterministic_offset', action="store_true", default=False)
    parser.add_argument('--deterministic_num_words', action="store_true", default=False)
    parser.add_argument('--deterministic_hw_size', action="store_true", default=False)

    parser.add_argument('--deterministic_angle', action="store_true", default=False)
    parser.add_argument('--deterministic_vertical', action="store_true", default=False)
    parser.add_argument('--deterministic_horizontal', action="store_true", default=False)

    parser.add_argument('--deterministic_rot', action="store_true", default=False)
    parser.add_argument('--deterministic_noise', action="store_true", default=False)
    parser.add_argument('--deterministic_crop', action="store_true", default=False)
    parser.add_argument('--deterministic_temp', action="store_true", default=False)

    parser.add_argument('--fully_random', action='store_true', default=False)
    parser.add_argument('--hw_text', action='store_true', default=False)
    parser.add_argument('--wrinkles', action='store_true', default=False)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--bbox', action='store_true', default=False)

    return parser


def writeCSV(args):
    csv_file_path = os.path.join(args.output_directory, 'Coordinates.csv')
    if os.path.isfile(csv_file_path) == False:
        with open(csv_file_path, 'a') as ground_truth_file:
            writer = csv.writer(ground_truth_file)
            if args.start_index != -1:
                writer.writerow(["Filename", "class", "x_center", "y_center", "width", "height"])

    grid_file_path = os.path.join(args.output_directory, 'gridsizes.csv')
    if os.path.isfile(grid_file_path) == False:
        with open(grid_file_path, 'a') as gridsize_file:
            writer = csv.writer(gridsize_file)
            if args.start_index != -1:
                writer.writerow(["filename", "xgrid", "ygrid", "lead_name", "start", "end"])


def newwrinkles(style_path, cont_path, output_path, ifaddback):
    bimg = cv2.imread(style_path)
    if "white" not in output_path:
        cv2.imwrite(output_path[:-4] + "style.png", bimg)
    bgray = cv2.cvtColor(bimg, cv2.COLOR_BGR2GRAY).astype(int)

    fimg0 = cv2.imread(cont_path)

    fimg = cv2.resize(fimg0, dsize=bgray.shape, interpolation=cv2.INTER_NEAREST).astype(int)

    Y, X, N = fimg.shape
    exfimg = np.zeros((Y + 26, X + 26, N))
    exfimg[13:-13, 13:-13, :] = fimg  #
    for i in range(13):  #
        exfimg[13:-13, i, :] = fimg[:, 0, :]  #
        exfimg[13:-13, -i - 1, :] = fimg[:, -1, :]  #
    for i in range(13):  #
        exfimg[i, :, :] = exfimg[13, :, :]  #
        exfimg[-i - 1, :, :] = exfimg[-14, :, :]  #

    Y, X, N = fimg.shape
    exfimg = np.zeros((Y + 39, X + 39, N))
    exfimg[26:-13, 26:-13, :] = fimg  #
    for i in range(15):  #
        exfimg[26:-13, i, :] = fimg[:, 1, :]  #
        exfimg[26:-13, i + 13, :] = fimg[:, 1, :]  #
        exfimg[26:-13, -i - 1, :] = fimg[:, -1, :]  #
    for i in range(15):  #
        exfimg[i, :, :] = exfimg[28, :, :]  #
        exfimg[i + 13, :, :] = exfimg[28, :, :]  #
        exfimg[-i - 1, :, :] = exfimg[-14, :, :]  #

    goffset = (bgray - 128) / 10
    offsetLim1 = (np.floor(goffset) + 13).astype(int)
    offsetLim2 = (np.ceil(goffset) + 13).astype(int)
    sep1 = (goffset - np.floor(goffset)).flatten()
    sep2 = (np.ceil(goffset) - goffset).flatten()
    XX, YY = np.meshgrid(range(exfimg.shape[0] - 39), range(exfimg.shape[1] - 39))
    XX1, YY1 = XX + offsetLim1, YY + offsetLim1  #
    XX2, YY2 = XX + offsetLim2, YY + offsetLim2  #

    c1 = exfimg[YY1.flatten(), XX1.flatten(), :]
    c2 = exfimg[YY2.flatten(), XX2.flatten(), :]

    p1 = np.where(sep1 == 0, c1[:, 0], c2[:, 0] * sep1.flatten() + c1[:, 0] * sep2.flatten())
    p2 = np.where(sep1 == 0, c1[:, 1], c2[:, 1] * sep1.flatten() + c1[:, 1] * sep2.flatten())
    p3 = np.where(sep1 == 0, c1[:, 2], c2[:, 2] * sep1.flatten() + c1[:, 2] * sep2.flatten())

    newarr = np.array([p1.reshape(bgray.shape), p2.reshape(bgray.shape), p3.reshape(bgray.shape)])
    if ifaddback == 1:
        newarr = newarr * bgray / 255
    newarr = newarr.transpose((1, 2, 0)).astype(np.uint8)
    data = cv2.resize(newarr, dsize=(1100, 850))
    cv2.imwrite(output_path[:-4] + "wrinkles.png", data)


def run_single_file(args):
    sty = [0, 1]  # 0:default,1:rand
    stt = random.choice(sty)

    if stt == 0:
        args.random_dc = 0
        args.random_bw = 0
        args.random_grid_present = 1
        args.random_add_header = 0
        args.random_grid_color = False
    if stt == 1:
        args.random_dc = 0.5
        args.random_bw = 0
        args.random_grid_present = 1
        args.random_add_header = 0.5
        args.random_grid_color = True

    # if os.path.exists(args.output_directory + args.input_file[-30:-4] + "-0wrinkles.png"):
    #     return
    styles_path = "./styleimagenew/"
    styles_paths = os.listdir(styles_path)

    if hasattr(args, 'st') == True:
        random.seed(args.seed)

    filename = args.input_file
    header = args.header_file
    resolution = random.choice(range(50, args.resolution + 1)) if (args.random_resolution) else args.resolution
    padding = random.choice(range(0, args.pad_inches + 1)) if (args.random_padding) else args.pad_inches

    papersize = ''

    lead = args.deterministic_lead

    lead = True if bernoulli(0.5).rvs() == 1 else False

    bernoulli_dc = bernoulli(args.random_dc)
    bernoulli_bw = bernoulli(args.random_bw)
    bernoulli_grid = bernoulli(args.random_grid_present)
    if args.print_header:
        bernoulli_add_print = bernoulli(1)
    else:
        bernoulli_add_print = bernoulli(args.random_add_header)

    font = os.path.join('Fonts', random.choice(os.listdir("./Fonts")))

    if (args.random_bw == 0):
        if args.random_grid_color == False:
            standard_colours = args.standard_grid_color
        else:
            standard_colours = -1
    else:
        standard_colours = False

    data = [2, 4, 1]
    num_c = random.choice(data)

    if num_c == 1:
        ful_mode = 1
    else:
        ful_mode = 0

    style_number = random.randint(1, 15)
    style_path = styles_path + styles_paths[style_number]
    '''
    print(args.random_dc,args.random_bw,args.random_grid_present,args.random_add_header,args.random_grid_color)
    print("1:",style_path)
    print("2:",filename)
    print("3:",header)
    print("4:",args.start_index)
    print("5:",args.store_config)
    print("6:",args.store_text_bounding_box)
    print("8:",resolution)
    print("9:",papersize)
    print("10:",lead)
    print("11:",bernoulli_dc)
    print("12:",bernoulli_bw)
    print("13:",bernoulli_grid)
    print("14:",bernoulli_add_print)
    print("15:",padding)

    print("17:",standard_colours)
    print("18:",args.full_mode)
    print("19:",args.bbox)
    print("20:",num_c)
    print("21:",args.seed)
    print("22:",ful_mode)
    '''
    if stt == 0:
        # print("4:",args.start_index)
        args.store_config = True
        args.store_text_bounding_box = True

        # print("8:",resolution)
        # print("9:",papersize)
        # print("10:",lead)

        standard_colours = 5

        args.bbox = True
        num_c = 4

    if stt == 1:
        # print("4:",args.start_index)
        args.store_config = True
        args.store_text_bounding_box = True

        # print("8:",resolution)
        # print("9:",papersize)
        # print("10:",lead)

        args.bbox = True

    wrinkles = True
    out_array = get_paper_ecg(style_path, input_file=filename, header_file=header, start_index=args.start_index,
                              store_configs=args.store_config, store_text_bbox=args.store_text_bounding_box,
                              output_directory=args.output_directory, resolution=resolution, papersize=papersize,
                              add_lead_names=lead, add_dc_pulse=bernoulli_dc, add_bw=bernoulli_bw,
                              show_grid=bernoulli_grid, add_print=bernoulli_add_print, pad_inches=padding,
                              font_type=font, standard_colours=standard_colours, full_mode=args.full_mode,
                              bbox=args.bbox, columns=num_c, seed=args.seed, ful_mode=ful_mode)

    for out in out_array:
        if (args.fully_random):
            hw_text = random.choice((True, False))
            wrinkles = random.choice((True, False))
            augment = random.choice((True, False))
        else:
            hw_text = args.hw_text
            wrinkles = args.wrinkles
            augment = args.augment

        # Handwritten text addition
        if (hw_text):
            num_words = args.num_words if (args.deterministic_num_words) else random.choice(
                range(2, args.num_words + 1))
            x_offset = args.x_offset if (args.deterministic_offset) else random.choice(range(1, args.x_offset + 1))
            y_offset = args.y_offset if (args.deterministic_offset) else random.choice(range(1, args.y_offset + 1))

            out = get_handwritten(link=args.link, num_words=num_words, input_file=out, output_dir=args.output_directory,
                                  x_offset=x_offset, y_offset=y_offset,
                                  handwriting_size_factor=args.handwriting_size_factor, bbox=args.bbox)

        if (True):
            a = 1

            newwrinkles(style_path, out, out, 1)

            newwrinkles(style_path, out[:-4] + "white.png", out[:-4] + "white.png", 1)

            # ifWrinkles = True
            # ifCreases = True
            # crease_angle = args.crease_angle if (args.deterministic_angle) else random.choice(range(0,args.crease_angle+1))
            # num_creases_vertically = args.num_creases_vertically if (args.deterministic_vertical) else random.choice(range(1,args.num_creases_vertically+1))
            # num_creases_horizontally = args.num_creases_horizontally if (args.deterministic_horizontal) else random.choice(range(1,args.num_creases_horizontally+1))
            # out = get_creased(out,output_directory=args.output_directory,ifWrinkles=ifWrinkles,ifCreases=ifCreases,crease_angle=crease_angle,num_creases_vertically=num_creases_vertically,num_creases_horizontally=num_creases_horizontally,bbox = args.bbox)

        if (augment):
            noise = args.noise if (args.deterministic_noise) else random.choice(range(1, args.noise + 1))

            if (not args.bbox):
                do_crop = random.choice((True, False))
                if (do_crop):
                    crop = args.crop
                else:
                    crop = args.crop
            else:
                crop = 0
            blue_temp = random.choice((True, False))

            if (blue_temp):
                temp = random.choice(range(2000, 4000))
            else:
                temp = random.choice(range(10000, 20000))

            out = get_augment(out, output_directory=args.output_directory, rotate=args.rotate, noise=noise, crop=crop,
                              temperature=temp, bbox=args.bbox, store_text_bounding_box=args.store_text_bounding_box)

    return len(out_array)


if __name__ == '__main__':
    run_single_file(get_parser().parse_args(sys.argv[1:]))
