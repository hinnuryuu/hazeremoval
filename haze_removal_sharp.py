# 主逻辑
import sys

from cv2 import cv2 as cv  # 计算机视觉库主要函数库
import numpy as np  # 科学计算库
from PyQt5 import QtWidgets, QtCore  # PyQt5图形库
from cv2.ximgproc import l0Smooth

import window_function  # 连接界面和主逻辑的桥梁


# 盒式过滤,对所提供的图像进行平均模糊处理
def box_filter(img, r):
    (rows, cols) = img.shape  # 获得图像矩阵的行数和列数
    imDst = np.zeros_like(img)  # 构建一个与img同型的零矩阵

    imCum = np.cumsum(img, 0)  # 按行累加图像矩阵，换句话说，将矩阵中所有行全部加起来得到一行
    imDst[0: r + 1, :] = imCum[r: 2 * r + 1, :]  # 在第一行前r行中，计算r行的和，并赋值给第一行
    imDst[r + 1: rows - r, :] = imCum[2 * r + 1: rows, :] - imCum[0: rows - 2 * r - 1, :]  # 在第一行后r行中，计算r行的和，并赋值给第一行
    imDst[rows - r: rows, :] = \
        np.tile(imCum[rows - 1, :], [r, 1]) - imCum[rows - 2 * r - 1: rows - r - 1, :]  # 在最后一行前r行中，计算r行的和，并赋值给最后一行

    imCum = np.cumsum(imDst, 1)  # 按列累加图像矩阵，换句话说，将矩阵中所有列全部加起来得到一列
    imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]  # 在第一列前r列中，计算r列的和，并赋值给第一列
    imDst[:, r + 1: cols - r] = imCum[:, 2 * r + 1: cols] - imCum[:, 0: cols - 2 * r - 1]  # 在第一列后r列中，计算r列的和，并赋值给第一列
    imDst[:, cols - r: cols] = \
        np.tile(imCum[:, cols - 1], [r, 1]).T - imCum[:, cols - 2 * r - 1: cols - r - 1]  # 在最后一列前r列中，计算r列的和，并赋值给最后一列

    return imDst


# 导向滤波函数,滤波器p,在图像image的引导下,用r作为盒式滤波器的半径
def guided_filter(image, p, r, eps):
    (rows, cols) = image.shape  # 获得图像矩阵的行数和列数
    N = box_filter(np.ones([rows, cols]), r)  # 构建一个盒式滤波器

    meanI = box_filter(image, r) / N  # 计算图像的平均值
    meanP = box_filter(p, r) / N  # 计算引导图像的平均值
    meanIp = box_filter(image * p, r) / N  # 计算图像与引导图像的乘积的平均值
    covIp = meanIp - meanI * meanP  # 计算图像与引导图像的乘积的协方差

    meanII = box_filter(image * image, r) / N  # 计算图像的平方的平均值
    varI = meanII - meanI * meanI  # 计算图像的方差

    a = covIp / (varI + eps)  # 计算
    b = meanP - a * meanI  # 计算b

    meanA = box_filter(a, r) / N  # 计算a的平均值
    meanB = box_filter(b, r) / N  # 计算b的平均值

    q = meanA * image + meanB  # 计算图像的引导滤波结果
    return q


# 暗道处理生成
def gen_dark_channel(img, kernel):
    temp = np.amin(img, axis=2)  # 按照通道计算最小值
    return cv.erode(temp, kernel)  # 进行腐蚀操作


# 计算大气光值,在雾霾模型方程中计算大气组成
def calculate_atmosphere_light(img, channel_dark, top_percent):
    R, C, D = img.shape

    # 将黯淡部份压平,以获得亮点的最高比例。
    flat_dark = channel_dark.ravel()
    req = int((R * C * top_percent) / 100)
    indices = np.argpartition(flat_dark, -req)[-req:]

    # 找到黑暗中的最高强度指数
    flat_img = img.reshape(R * C, 3)
    return np.max(flat_img.take(indices, axis=0), axis=0)


# 计算透射率
def get_transmission(dark_div, param):
    transmission = 1 - param * dark_div
    return transmission


# 用经过导向滤波的图像生成深度图
def get_depth_map(trans, beta):
    rval = -np.log(trans) / beta
    return rval / np.max(rval)


# 在评估雾度的所有3个组成部分后得到结果去雾的图像
def get_radiant(image, atm, t, thres):  # 参数分别是原图像矩阵,大气光值,最终透射率和thres
    R, C, D = image.shape
    temp = np.empty(image.shape)  # 创建一个空白图像
    t[t < thres] = thres  # 将低于阈值的值设置为阈值
    for i in range(D):  # 遍历每一个通道
        temp[:, :, i] = t  # 将传输图赋值给空白图像
    b = (image - atm) / temp + atm  # 计算去雾图
    b[b > 255] = 255  # 将去雾图的值限制在0~255之间
    return b  # 返回去雾


# 给图像添加高斯噪声
def add_noise(img):
    img = img.astype(np.float32)  # 将图像转换为float32格式
    img = img + np.random.normal(0, 3, img.shape)  # 在图像上添加0~3的均值为0的高斯噪声
    img[img > 255] = 255  # 将噪声超过255的值限制在255以内
    img[img < 0] = 0  # 将噪声小于0的值限制在0以内
    return img.astype(np.uint8)  # 将图像转换为uint8格式并返回


# 图像矩阵平移函数
def shift_image(image, direction):
    temp_image = image.copy()
    [m, n] = temp_image.shape
    if direction == 1:  # up
        temp_image[0:m - 1, :] = image[1:m, :]
    if direction == 2:  # left
        temp_image[:, 0:n - 1] = image[:, 1:n]
    if direction == 3:  # down
        temp_image[1:m, :] = image[0:m - 1, :]
    if direction == 4:  # right
        temp_image[:, 1:n] = image[:, 0:n - 1]
    return temp_image


# 全变分降噪函数
def total_variation(img, lamb):
    result = img.copy()
    img = np.float32(img)
    result = np.float32(result)
    for i in range(20):  # 迭代20次
        for j in range(3):  # 遍历三个通道
            u = result[:, :, j]
            img_temp_1 = shift_image(u, 1)  # 上移图像
            img_temp_2 = shift_image(u, 2)  # 左移图像
            img_temp_3 = shift_image(u, 3)  # 下移图像
            img_temp_4 = shift_image(u, 4)  # 右移图像
            u_normal = ((img_temp_4 - u) ** 2 + (img_temp_3 - u) ** 2) ** 0.5  # 计算上下左右图像的差值
            u = (img[:, :, j] * u_normal + lamb * (img_temp_1 + img_temp_2 + img_temp_3 + img_temp_4)) \
                / (1 * u_normal + 4 * lamb)  # 全变分降噪
            result[:, :, j] = u
    return result


# 计算归一化图像的梯度幅值作为权重
def get_grad_weight(img) -> float:
    img_x = cv.Scharr(img, cv.CV_64F, dx=1, dy=0)
    img_y = cv.Scharr(img, cv.CV_64F, dx=0, dy=1)
    grad_magnitude = cv.addWeighted(img_x, 0.5, img_y, 0.5, 0)  # 计算梯度的幅值
    grad_weight = np.sum(grad_magnitude) / (img.shape[0] * img.shape[1])  # 计算幅值的权重
    return grad_weight


# 从提供的雾图中计算清晰的图像
def remove_haze(original_img, top_percent=0.1,
                thres_haze=0.1, beta=1.0, eps=0.001,
                window_size=50, guided_filter_radius=75, remove_level=0.95,
                atmosphere_light_value=200, auto_atmosphere=True, multi_transmittance=False):
    if not multi_transmittance:  # 不使用多尺度透射率优化
        img_gray = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
        img = np.asarray(original_img, dtype=np.float64)  # 图像矩阵化
        img_normal = (img_gray - img_gray.mean()) / (img_gray.max() - img_gray.min())  # 归一化
        kernel = np.ones((window_size, window_size), np.float64)  # 定义最小滤波器边长
        dark = gen_dark_channel(img, kernel)  # 计算暗通道

        A = calculate_atmosphere_light(img, dark, top_percent) if auto_atmosphere else atmosphere_light_value

        dark_div = gen_dark_channel(img / A, kernel)  # 计算大气后的暗通道
        # 以下为单尺度返回值
        t_estimate = get_transmission(dark_div, remove_level)  # 计算初步透射率
        t_refined = guided_filter(img_normal, t_estimate, guided_filter_radius, eps)  # 计算最终透射率
        unhazed = total_variation(add_noise(get_radiant(img, A, t_refined, thres_haze)), 0.1)  # 计算降噪去雾图
        depth_map = get_depth_map(t_refined, beta)  # 计算深度图(可选计算)
        return [np.array(x, dtype=np.uint8) for x in
                [t_estimate * 255, t_refined * 255,
                 unhazed, dark, depth_map * 255]]

    else:  # 使用多尺度透射率优化
        img_gray = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
        img_normal = (img_gray - img_gray.mean()) / (img_gray.max() - img_gray.min())  # 归一化
        cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
        img = np.asarray(original_img, dtype=np.float64)  # 图像矩阵化
        point_window_size = 1  # 点估计尺度
        area_window_size = window_size  # 块估计尺度
        point_kernel = np.ones((point_window_size, point_window_size), np.float64)
        area_kernel = np.ones((area_window_size, area_window_size), np.float64)
        point_dark = gen_dark_channel(img, point_kernel)  # 计算点估计暗通道图
        area_dark = gen_dark_channel(img, area_kernel)  # 计算块估计暗通道图
        point_A = calculate_atmosphere_light(img, point_dark,
                                             top_percent) if auto_atmosphere else atmosphere_light_value
        area_A = calculate_atmosphere_light(img, area_dark,
                                            top_percent) if auto_atmosphere else atmosphere_light_value  # 作为图像的大气值估计带入
        point_dark_div = gen_dark_channel(img / point_A, point_kernel)  # 经过天空正则化后的点暗通道图
        area_dark_div = gen_dark_channel(img / area_A, area_kernel)  # 经过天空正则化后的块暗通道图
        point_estimate = get_transmission(point_dark_div, remove_level)  # 点估计透射率图
        area_estimate = get_transmission(area_dark_div, remove_level)  # 块估计透射率图
        smooth_dst = l0Smooth(point_estimate, lambda_=0.01, kappa=2)  # 点估计初步透射平滑滤波
        '''
        gaussian_window_size = 3  # 高斯滤波窗口大小
        gaussian_dst = cv2.GaussianBlur(area_estimate, (gaussian_window_size, gaussian_window_size),
                                        3 * ((gaussian_window_size - 1) * 0.5 - 1) + 0.8)  # 块估计初步透射高斯滤波
        '''
        gaussian_dst = guided_filter(img_normal, area_estimate, guided_filter_radius, eps)  # 引导滤波代替高斯滤波处理
        point_weight = get_grad_weight(point_estimate)  # 点归一化后的梯度幅值作为融和的权重w1
        area_weight = get_grad_weight(area_estimate)  # 块归一化的梯度幅值作为融和的权重w2
        point_weight = point_weight / (point_weight + area_weight)
        area_weight = 1 - point_weight
        img_mixed = cv.addWeighted(smooth_dst, point_weight, gaussian_dst, area_weight, 0)  # 图像加权融和
        # print(point_weight, area_weight)  # 限定调试阶段输出，点权重和块权重
        # unhazed = total_variation(add_noise(get_radiant(img, area_A, img_mixed, thres_haze)), 0.1)  # 计算降噪去雾图
        unhazed = get_radiant(img, area_A, img_mixed, thres_haze)
        return [point_estimate * 255, area_estimate * 255,
                point_dark, area_dark,
                smooth_dst * 255, gaussian_dst * 255,
                img_mixed * 255, unhazed]


# 降噪函数封装
def denoise_for_img(img):
    return total_variation(add_noise(img), 2)


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 俺电脑是2K高分屏，不开这个会错乱排列
    app = QtWidgets.QApplication(sys.argv)
    main_window = window_function.MainGUI()
    main_window.show()
    sys.exit(app.exec_())
