##############################################################################
# FILE: cartoonify.py
# WRITERS: Dana Aviran, 211326608, dana.av
# EXERCISE: Intro2cs2 ex6 2021-2022
# DESCRIPTION: manipulation on RGB images
##############################################################################

import ex6_helper
from ex6_helper import *
from typing import Optional
import math
import sys


def separate_channels(image: ColoredImage) -> List[List[List[int]]]:
    # this function takes a colored three dimensional list image and
    # turns it into a list if separated channels
    height = len(image)
    width = len(image[0])
    num_of_channels = len(image[0][0])
    separated_list = []
    current_row = []
    current_channel = []
    i, j, s = 0, 0, 0
    while s < num_of_channels:
        while i < height:
            while j < width:
                # we append the current pixel value of current channel
                current_row.append(image[i][j][s])
                # we go through all pixels in row
                j += 1
            # we append the row of current values to the channel
            current_channel.append(current_row)
            current_row = []  # we restart the value of current row
            # we go to the next row
            j = 0
            i += 1
        # when we finish all rows, we append the list of rows of current
        # channel to the list of all channels
        separated_list.append(current_channel)
        # we restart the current channel value
        current_channel = []
        i = 0
        # and we go to next round - the next channel
        s += 1
    # at last, we return the separated list
    return separated_list


def combine_channels(channels: List[List[List[int]]]) -> ColoredImage:
    # this function gets a list of separated channels and combines them
    # into a three dimensional colored image
    i, j, s = 0, 0, 0
    current_row = []
    current_pixel = []
    color_image = []
    num_of_channels = len(channels)
    num_of_rows_in_channel = len(channels[0])
    num_of_pixels_in_row = len(channels[0][0])
    # we go through all the channels, taking to the current pixel all the
    # values of same indexes
    while j < num_of_rows_in_channel:
        while s < num_of_pixels_in_row:
            while i < num_of_channels:
                current_pixel.append(channels[i][j][s])
                i += 1
            current_row.append(current_pixel)
            current_pixel = []
            i = 0
            s += 1
        color_image.append(current_row)
        current_row = []
        s = 0
        j += 1
    return color_image


def RGB2grayscale(colored_image: ColoredImage) -> SingleChannelImage:
    # this function gets a colored image and returns a two dimensional
    # black and white image
    height = len(colored_image)
    width = len(colored_image[0])
    i, j = 0, 0
    current_row = []
    bw_image = []
    while i < height:
        while j < width:
            current_value = round(colored_image[i][j][0] * 0.299 + \
                                  colored_image[i][j][1] * 0.587 + \
                                  colored_image[i][j][2] * 0.114)
            current_row.append(current_value)
            j += 1
        bw_image.append(current_row)
        current_row = []
        j = 0
        i += 1
    return bw_image


def blur_kernel(size: int) -> Kernel:
    # this function returns a kernel with height and width as in size
    i, j = 0, 0
    current_row = []
    kernel = []
    while i < size:
        while j < size:
            current_row.append(1 / (size ** 2))  # as written in instructions
            j += 1
        kernel.append(current_row)
        current_row = []
        j = 0
        i += 1
    return kernel


def apply_kernel(image: SingleChannelImage,
                 kernel: Kernel) -> SingleChannelImage:
    # this function gets a single channel image and a kernel, and blurs
    # the image with a formula written in the instructions
    i, j = 0, 0
    kernel_len = len(kernel)
    kernel_mid_len = int((kernel_len - 1) / 2)  # the r value of the kernel
    height = len(image)
    width = len(image[0])
    calculate = 0
    new_image = deepcopy(image)
    # we go over all the pixels
    while i < height:
        while j < width:
            if i >= kernel_mid_len:
                # if i is larger than than kernel middle length
                # the overlap upwards is the middle length
                overlap_up = kernel_mid_len
            else:
                # else, i is smaller than the kernel middle length
                # and we set the overlap to be i
                overlap_up = i
            if (height - 1 - i) >= kernel_mid_len:
                overlap_down = kernel_mid_len
            # if length of image minus i is larger than kernel's mid length
            # the overlap left is the middle length
            else:
                # else, it is smaller than the kernel middle length
                # and we set the overlap to be it
                overlap_down = height - i - 1
            if j >= kernel_mid_len:
                # if j is larger than kernel's mid length
                # the overlap left is the middle length
                overlap_left = kernel_mid_len
            else:
                # else, it is smaller than the kernel middle length
                # and we set the overlap to be it
                overlap_left = j
            if (width - 1 - j) >= kernel_mid_len:
                # if width of image minus j is larger than kernel's mid length
                # the overlap right is the middle length
                overlap_right = kernel_mid_len
            else:
                # else, it is smaller than the kernel middle length
                # and we set the overlap to be it
                overlap_right = width - 1 - j
            # the height of the overlap square of kernel in image
            overlap_square_length = overlap_up + overlap_down + 1
            # the width of the overlap square of kernel in image
            overlap_square_width = overlap_right + overlap_left + 1
            # the area of the overlap square of kernel in image
            overlap_square_area = (overlap_square_width *
                                   overlap_square_length)
            # the index of row of the first pixel in the overlap square
            # while the loop will progress, it won't be first anymore
            row_index = i - overlap_up
            # the index of column of the first pixel in the overlap square
            # while the loop will progress, it won't be first anymore
            column_index = j - overlap_left
            # we set counters for the loop that are different from the pixel
            # indexes:
            counter_row = 0
            counter_column = 0
            # we start the while loop
            while counter_row < overlap_square_length:
                while counter_column < overlap_square_width:
                    # we add the value of the kernel applied on the pixel
                    calculate += (image[row_index][
                                    column_index] / (kernel_len ** 2))
                    column_index += 1
                    counter_column += 1
                column_index = j - overlap_left
                counter_column = 0
                row_index += 1
                counter_row += 1
            # we calculate the amount of pixels in kernel that did not overlap
            remaining_kernel = ((kernel_len ** 2) - overlap_square_area)
            # we add them to the sum as given in instructions
            calculate += (remaining_kernel * image[i][j]) / (kernel_len ** 2)
            if calculate > 255:
                calculate = 255
            if calculate < 0:
                calculate = 0
            # we insert the sum value of counter in the current pixel
            new_image[i][j] = round(calculate)
            # we restart the counter and continue the loop
            calculate = 0
            j += 1
        j = 0
        i += 1
    # we return the blurred image
    return new_image


def bilinear_interpolation(image: SingleChannelImage, y: float,
                           x: float) -> int:
    # this function gets a single channel image and the coordinates of a pixel
    # as it falls in the original image and calculates a value that will be
    # inserted in a the suiting pixel in the new image
    if type(y) is int:
        delta_y = 0  # difference between the roundings and the value is zero
    else:
        delta_y = math.ceil(y) - y  # else,the difference is calculated like so
    if type(x) is int:
        delta_x = 0  # difference between the roundings and the value is zero
    else:
        delta_x = math.ceil(x) - x  # else,the difference is calculated like so
    # we check the values of 4 pixels around the current pixel
    a = image[math.floor(y)][math.floor(x)]
    b = image[math.ceil(y)][math.floor(x)]
    c = image[math.floor(y)][math.ceil(x)]
    d = image[math.ceil(y)][math.ceil(x)]
    # we calculate the sum as in the instructions of bilinear interpolation
    calculate_a = a * (1 - delta_x) * (1 - delta_y)
    calculate_b = b * (1 - delta_x) * delta_y
    calculate_c = c * delta_x * (1 - delta_y)
    calculate_d = d * delta_x * delta_y
    return round(calculate_a + calculate_b + calculate_c + calculate_d)


def make_image_pattern(height, width):
    # this function gets values of height and width and creates a single
    # channel image in those dimensions while the value of every pixel is zero
    i, j = 0, 0
    image_pattern = []
    while i < height:
        image_pattern.append([])
        i += 1
    i = 0
    while i < height:
        while j < width:
            image_pattern[i].append(0)
            j += 1
        j = 0
        i += 1
    return image_pattern


def resize(image: SingleChannelImage, new_height: int,
           new_width: int) -> SingleChannelImage:
    # this function gets a single channel image and values of new height and
    # width and resizes the image to the new dimensions while calculating
    # each pixel value with the bilinear interpolation method
    original_height = len(image)
    original_width = len(image[0])
    # we calculate the ratios between original image to new one
    ratio_height = original_height / new_height
    ratio_width = original_width / new_width
    # we make the pattern of new image
    new_image = make_image_pattern(new_height, new_width)
    i, j = 0, 0
    while i < new_height:
        while j < new_width:
            # we calculate the pixel indexes in the original image
            y = i * ratio_height
            x = j * ratio_width
            # if proceed the limits of dimensions we fix that
            if y > original_height - 1:
                y = original_height - 1
            if x > original_width - 1:
                x = original_width - 1
            # we calculate the value using the bilinear function and insert
            # it to the pixel in the new image
            new_image[i][j] = bilinear_interpolation(image, y, x)
            # if it proceeds the limits, we fix that
            if new_image[i][j] > 255:
                new_image[i][j] = 255
            if new_image[i][j] < 0:
                new_image[i][j] = 0
            j += 1
        j = 0
        i += 1
    # we insert the values in the corners as same in the original image
    new_image[0][0] = image[0][0]
    new_image[0][new_width - 1] = image[0][original_width - 1]
    new_image[new_height - 1][0] = image[original_height - 1][0]
    new_image[new_height - 1][new_width - 1] = \
        image[original_height - 1][original_width - 1]
    # at last, we return the new image
    return new_image


def scale_down_colored_image(image: ColoredImage, max_size: int) -> Optional[
    ColoredImage]:
    # this function gets a colored image and a maximum size and returns a
    # smaller version of the image, as the height and width are smaller than
    # the maximum size, and while the proportions of the image are kept
    height = len(image)
    width = len(image[0])
    num_of_channels = len(image[0][0])
    # if the photo is inside the max_size square
    if height <= max_size and width <= max_size:
        return None
    # else, we separate the list to different channels
    separate_list = separate_channels(image)
    new_separate_list = []
    # if there are more rows than pixels in each row
    if height >= width:
        i = 0
        # we set the bigger height of the new image to be the max size
        new_height = max_size
        # we set the ratio as smaller than 1
        ratio = width / height
        # we set the width value and check if it can be rounded up
        new_width = new_height * ratio
        # if it can't be, we round it down
        if math.ceil(new_width) > max_size:
            new_width = math.floor(new_width)
        else:
            new_width = math.ceil(new_width)
        # the new one channel image height is the max_size,
        # so we insert the new resized list by using the resize function
        # on the new height and width values
        while i < num_of_channels:
            separate_list[i] = resize(separate_list[i], max_size,
                                      new_width)
            new_separate_list.append(separate_list[i])
            i += 1
    # else, there are more pixels in row than rows,
    # we set the width of the new image to be the max size
    else:
        i = 0
        # we set the width of the new image to be the max size
        new_width = max_size
        # we set the ratio differently so it will be smaller than one
        ratio = height / width
        # we set the new height value using the ratio we calculated
        new_height = new_width * ratio
        # and check if it can be rounded up
        if math.ceil(new_height) > max_size:
            new_height = math.floor(new_height)
        else:
            new_height = math.ceil(new_height)
        # the new one channel image height is the y_value,
        # so we insert the new resized list by using the resize function
        # on the new height and width values
        while i < num_of_channels:
            separate_list[i] = resize(separate_list[i], new_height, max_size)
            new_separate_list.append(separate_list[i])
            i += 1
    # we merge together all the channels and return the new resized image
    new_separate_list = combine_channels(new_separate_list)
    # and at last, we return the combined list
    return new_separate_list


def is_colored_image(image):
    # this function checks if the image in colored or single channel
    # and return a boolean value in return
    if type(image[0][0]) is int:
        return False
    else:
        return True


def rotate_90(image: Image, direction: str) -> Image:
    # this function gets an image and a char of a direction and returns
    # the image rotated to that direction
    original_height = len(image)
    original_width = len(image[0])
    new_image = []
    new_row = []
    # we check if there are more than one channel
    is_colored = is_colored_image(image)
    # if it is a single channel image
    if not is_colored:
        # right
        if direction == 'R':
            i = original_height - 1
            j = 0
            while j < original_width:
                while i >= 0:
                    new_row.append(image[i][j])
                    i -= 1
                new_image.append(new_row)
                new_row = []
                i = original_height - 1
                j += 1
        # left
        elif direction == 'L':
            i = 0
            j = original_width - 1
            while j >= 0:
                while i < original_height:
                    new_row.append(image[i][j])
                    i += 1
                new_image.append(new_row)
                new_row = []
                i = 0
                j -= 1
    else:
        # right
        if direction == 'R':
            i = original_height - 1
            j = 0
            while j < original_width:
                while i >= 0:
                    new_row.append(image[i][j])
                    i -= 1
                new_image.append(new_row)
                new_row = []
                i = original_height - 1
                j += 1
        # left
        if direction == 'L':
            i = 0
            j = original_width - 1
            while j >= 0:
                while i < original_height:
                    new_row.append(image[i][j])
                    i += 1
                new_image.append(new_row)
                new_row = []
                i = 0
                j -= 1
    return new_image


def get_edges(image: SingleChannelImage, blur_size: int, block_size: int,
              c: int) -> SingleChannelImage:
    # this function gets a single channel image, a value for the kernel to
    # blur the image, a block size for the kernel to calculate the threshold
    # and a constant to subtract from the threshold
    # first, we blur the image by using the apply kernel function
    blurred_image = apply_kernel(image, blur_kernel(blur_size))
    i, j = 0, 0
    height = len(image)
    width = len(image[0])
    # we make a kernel
    kernel = blur_kernel(block_size)
    # we insert the value of 1 to all the pixels in the kernel
    while i < block_size:
        while j < block_size:
            kernel[i][j] = 1
            j += 1
        j = 0
        i += 1
    # we take the blurred image and insert in its pixels new values, as
    # calculated in the function apply_kernel
    threshold = apply_kernel(blurred_image, kernel)
    i, j = 0, 0
    # we go over each pixel value and check if it is bigger or smaller than
    # threshold and insert the values of 255 and 0 accordingly
    while i < height:
        while j < width:
            # we also subtract the constant from the threshold
            threshold[i][j] -= c
            if blurred_image[i][j] > threshold[i][j]:
                blurred_image[i][j] = 255
            else:
                blurred_image[i][j] = 0
            j += 1
        j = 0
        i += 1
    # at last, we return the new image
    return blurred_image


def quantize(image: SingleChannelImage, N: int) -> SingleChannelImage:
    # this function gets a single channel image and a constant N  value
    # and returns a quantized image (less colors)
    i, j = 0, 0
    height = len(image)
    width = len(image[0])
    new_image = make_image_pattern(height, width)
    while i < height:
        while j < width:
            new_image[i][j] = round(math.floor(image[i][j] * (N / 256)) *
                                    (255 / (N - 1)))
            j += 1
        j = 0
        i += 1
    return new_image


def quantize_colored_image(image: ColoredImage, N: int) -> ColoredImage:
    # this function gets a colored image and a constant N value and returns
    # a quantized image
    separated_channels = separate_channels(image)
    i = 0
    height = len(separated_channels)
    # we apply the quantize function on each sub channel
    while i < height:
        separated_channels[i] = quantize(separated_channels[i], N)
        i += 1
    # we combine the channels back together
    new_image = combine_channels(separated_channels)
    return new_image


def add_mask(image1: Image, image2: Image, mask: List[List[float]]) -> Image:
    # this function gets two images - both colored or both single channeled
    # and a mask - a single channel image that is suited in dimensions to the
    # pixels of the two images, and merges the images together as calculated
    # by instructions
    height = len(image1)
    width = len(image1[0])
    new_image = make_image_pattern(height, width)
    i, j = 0, 0
    # if the image is a single channel image
    if not is_colored_image(image1):
        while i < height:
            while j < width:
                # we calculate the value of the new image pixel by instructions
                calculate = round((image1[i][j] * mask[i][j]) +
                                  (image2[i][j] * (1 - mask[i][j])))
                if calculate > 255:
                    new_image[i][j] = 255
                elif calculate < 0:
                    new_image[i][j] = 0
                else:
                    new_image[i][j] = calculate
                j += 1
            j = 0
            i += 1
    # else, if the image is colored
    else:
        # we separate both images to separate channels
        separated_image1 = separate_channels(image1)
        separated_image2 = separate_channels(image2)
        # we state the minimum number of channels
        if len(separated_image1) < len(separated_image2):
            channels_num = len(separated_image1)
        else:
            channels_num = len(separated_image2)
        # we make a new empty list of channels
        new_separated_image = make_image_pattern(channels_num, 1)
        s = 0
        # we fill each sub list in the list with the fitting dimensions
        while s < channels_num:
            new_separated_image[s] = make_image_pattern(height, width)
            while i < height:
                while j < width:
                    # we calculate each value of pixel by instructions
                    calculate = round(separated_image1[s][i][j]
                                      * mask[i][j] +
                                      separated_image2[s][i][j]
                                      * (1 - mask[i][j]))
                    if calculate > 255:
                        new_separated_image[s][i][j] = 255
                    elif calculate < 0:
                        new_separated_image[s][i][j] = 0
                    else:
                        new_separated_image[s][i][j] = calculate
                    j += 1
                j = 0
                i += 1
            i = 0
            s += 1
        # we combine the channels
        new_image = combine_channels(new_separated_image)
    # and return the new image
    return new_image


def make_mask(image):
    # this function gets a single channel image and makes a mask out of it
    # by divide each pixel value with 255
    height = len(image)
    width = len(image[0])
    i, j = 0, 0
    while i < height:
        while j < width:
            image[i][j] = image[i][j] / 255
            j += 1
        j = 0
        i += 1
    return image


def cartoonify(image: ColoredImage, blur_size: int, th_block_size: int,
               th_c: int, quant_num_shades: int) -> ColoredImage:
    # this function gets a colored image, a int value for the dimensions of
    # blur kernel, a value for the block size fot the kernel to be used in
    # quantize function, a constant th_c to reduce from the threshold that is
    # calculated in the quantize function, and a constant for the number of
    # shades, also for the quantize function
    height = len(image)
    width = len(image[0])
    # we quantize the colored image
    quantized = quantize_colored_image(image, quant_num_shades)
    # we get the edges of image
    edges = get_edges(RGB2grayscale(image), blur_size, th_block_size, th_c)
    # we get a separated channel of quantized
    sep_quantized = separate_channels(quantized)
    # we make a mask out of edges
    mask_edges = deepcopy(edges)
    for i in range(len(edges)):
        for j in range(len(edges[0])):
            mask_edges[i][j] = edges[i][j] / 255
    # now we go through all the channels in sep_quantized
    for i in range(len(sep_quantized)):
        sep_quantized[i] = add_mask(sep_quantized[i], edges, mask_edges)
    # we combine the channels and finely returned the cartooned image
    new_image = combine_channels(sep_quantized)
    return new_image


def main():
    # the main function of the program
    # if there are more or less arguments than should be, we print a message
    # and stop the program
    if len(sys.argv) != 8:
        print("invalid num of arguments")
    # we insert the values of arguments to fitting variables
    image_source = sys.argv[1]
    cartoon_dest = sys.argv[2]
    max_im_size = sys.argv[3]
    blur_size = sys.argv[4]
    th_block_size = sys.argv[5]
    th_c = sys.argv[6]
    quant_num_shades = sys.argv[7]
    # we load the image by using its source
    image = ex6_helper.load_image(image_source)
    height = len(image)
    width = len(image[0])
    # we check if one of the dimensions of image is bigger than max size and
    # if it is, we use the scale_down function to make it smaller
    if height > max_im_size or width > max_im_size:
        smaller_image = scale_down_colored_image(image, max_im_size)
    # we make the image a cartooned version using the arguments
    new_image = cartoonify(smaller_image, blur_size, th_block_size, th_c,
                       quant_num_shades)
    # lastly, we save the image in the destination
    ex6_helper.save_image(new_image, cartoon_dest)


if __name__ == '__main__':
    main()
