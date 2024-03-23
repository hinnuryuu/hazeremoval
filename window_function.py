# 界面与逻辑的联结
import os

import matplotlib.pyplot as plt  # 科学绘图库
from PyQt5.QtGui import QPixmap  # PyQt5图形库
from PyQt5.QtWidgets import QFileDialog, QMessageBox  # PyQt5图形库

import window  # 已经设计好的图形界面
from haze_removal_sharp import *  # 逻辑改进


# 注册图形界面类,功能有二
# 1.将图形界面与功能函数建立联系
# 2.设置图形界面中的控件功能
class MainGUI(QtWidgets.QMainWindow, QtWidgets.QWidget, window.Ui_MainWindow):
    def __init__(self) -> None:
        QtWidgets.QMainWindow.__init__(self)
        QtWidgets.QWidget.__init__(self)
        window.Ui_MainWindow.__init__(self)
        self.have_unhazed = None  # 已经去雾了吗?
        self.have_denoise = None  # 已经去噪了吗?
        self.multi_check_status = None  # 去雾方式
        self.filename = None  # 全局图像路径
        self.setupUi(self)

        self.lineEdit_1.setText(str(self.horizontalSlider_1.value()))
        self.lineEdit_2.setText(str(self.horizontalSlider_2.value()))
        self.lineEdit_3.setText(str(self.horizontalSlider_3.value() / 100))
        self.lineEdit_4.setText(str(self.horizontalSlider_4.value()))
        self.horizontalSlider_1.valueChanged.connect(self.change_slider_1)
        self.horizontalSlider_2.valueChanged.connect(self.change_slider_2)
        self.horizontalSlider_3.valueChanged.connect(self.change_slider_3)
        self.horizontalSlider_4.valueChanged.connect(self.change_slider_4)
        self.lineEdit_1.textChanged.connect(self.change_text_1)
        self.lineEdit_2.textChanged.connect(self.change_text_2)
        self.lineEdit_3.textChanged.connect(self.change_text_3)
        self.lineEdit_4.textChanged.connect(self.change_text_4)

        self.loadpicButton.clicked.connect(self.load_picture)
        self.savepicButton.clicked.connect(self.save_picture)
        self.defoggingButton.clicked.connect(self.defogging)
        self.exitButton.clicked.connect(self.close_window)
        self.histogramButton.clicked.connect(self.view_histogram)
        self.denoiseButton.clicked.connect(self.denoise)

        self.autoCheck.stateChanged.connect(self.auto_atmosphere)
        self.multiCheck.stateChanged.connect(self.multi_transmittance)

        self.savepicButton.setEnabled(False)  # 先锁住防止还未去雾就保存图片了
        self.histogramButton.setEnabled(False)  # 先锁住防止还未载入图片就显示直方图数据了
        self.defoggingButton.setEnabled(False)  # 先锁住防止还未载入图片就去雾了
        self.denoiseButton.setEnabled(False)  # 锁住降噪按钮
        self.autoCheck.setChecked(True)  # 默认启用自动大气光值
        self.multiCheck.setChecked(False)  # 默认禁用多尺度透射率

    # 以下8个函数是为数据同步实现文本框和滑条联动,功能有三
    # 1.逻辑实现保证滑条与文本框中的数值实时联动
    # 2.过滤了无效参数,例如负数,字母,符号等非数值类型
    # 3.确保了文本框中的数据一定存在,为后期的逻辑处理节省代码量

    # 当滑条改变时,去联动文本框
    def change_slider_1(self) -> None:
        self.lineEdit_1.setText(str(self.horizontalSlider_1.value()))

    def change_slider_2(self) -> None:
        self.lineEdit_2.setText(str(self.horizontalSlider_2.value()))

    def change_slider_3(self) -> None:
        self.lineEdit_3.setText(str(self.horizontalSlider_3.value() / 100))

    def change_slider_4(self) -> None:
        self.lineEdit_4.setText(str(self.horizontalSlider_4.value()))

    # 当文本框改变时,去联动滑条
    def change_text_1(self) -> None:
        try:
            value = int(self.lineEdit_1.text())
        except ValueError:
            value = self.horizontalSlider_1.minimum()
            self.lineEdit_1.setText(str(self.horizontalSlider_1.minimum()))
        if self.horizontalSlider_1.minimum() <= value <= self.horizontalSlider_1.maximum():
            self.horizontalSlider_1.setValue(value)
        else:
            if self.horizontalSlider_1.minimum() > value:
                self.horizontalSlider_1.setValue(self.horizontalSlider_1.minimum())
                self.lineEdit_1.setText(str(self.horizontalSlider_1.minimum()))
            else:
                self.horizontalSlider_1.setValue(self.horizontalSlider_1.maximum())
                self.lineEdit_1.setText(str(self.horizontalSlider_1.maximum()))

    def change_text_2(self) -> None:
        try:
            value = int(self.lineEdit_2.text())
        except ValueError:
            value = self.horizontalSlider_2.minimum()
            self.lineEdit_2.setText(str(self.horizontalSlider_2.minimum()))
        if self.horizontalSlider_2.minimum() <= value <= self.horizontalSlider_2.maximum():
            self.horizontalSlider_2.setValue(value)
        else:
            if self.horizontalSlider_2.minimum() > value:
                self.horizontalSlider_2.setValue(self.horizontalSlider_2.minimum())
                self.lineEdit_2.setText(str(self.horizontalSlider_2.minimum()))
            else:
                self.horizontalSlider_2.setValue(self.horizontalSlider_2.maximum())
                self.lineEdit_2.setText(str(self.horizontalSlider_2.maximum()))

    def change_text_3(self) -> None:
        try:
            value = int(round(float(self.lineEdit_3.text()), 2) * 100)  # 滑条不支持小数,所以考虑'曲线救国'--转为整数处理后还原
        except ValueError:
            value = self.horizontalSlider_3.minimum()
            self.lineEdit_3.setText(str(self.horizontalSlider_3.minimum()))
        if self.horizontalSlider_3.minimum() <= value <= self.horizontalSlider_3.maximum():
            self.horizontalSlider_3.setValue(value)
        else:
            if self.horizontalSlider_3.minimum() > value:
                self.horizontalSlider_3.setValue(self.horizontalSlider_3.minimum())
                self.lineEdit_3.setText(str(self.horizontalSlider_3.minimum() / 100))
            else:
                self.horizontalSlider_3.setValue(self.horizontalSlider_3.maximum())
                self.lineEdit_3.setText(str(self.horizontalSlider_3.maximum() / 100))

    def change_text_4(self) -> None:
        try:
            value = int(self.lineEdit_4.text())
        except ValueError:
            value = self.horizontalSlider_4.minimum()
            self.lineEdit_4.setText(str(self.horizontalSlider_4.minimum()))
        if self.horizontalSlider_4.minimum() <= value <= self.horizontalSlider_4.maximum():
            self.horizontalSlider_4.setValue(value)
        else:
            if self.horizontalSlider_4.minimum() > value:
                self.horizontalSlider_4.setValue(self.horizontalSlider_4.minimum())
                self.lineEdit_4.setText(str(self.horizontalSlider_4.minimum()))
            else:
                self.horizontalSlider_4.setValue(self.horizontalSlider_4.maximum())
                self.lineEdit_4.setText(str(self.horizontalSlider_4.maximum()))

    # 以下为按钮的功能实现
    def load_picture(self) -> None:  # 图片加载按钮
        self.have_unhazed = False  # 去雾状态记录还原
        self.have_denoise = False  # 降噪状态记录还原
        if not self.filename:
            self.filename = QFileDialog.getOpenFileName(self, "图片选择", filter='图像文件 (*.jpg *.png *.gif *.bmp)')[0]
        else:
            filename = QFileDialog.getOpenFileName(self, "图片选择", filter='图像文件 (*.jpg *.png *.gif *.bmp)')[0]
            if filename:
                self.filename = filename
        pixmap = QPixmap(self.filename)
        self.originalView.setPixmap(pixmap)
        self.originalView.setScaledContents(True)
        if self.filename:
            self.histogramButton.setEnabled(True)  # 解锁,载入成功可以显示直方图数据了
            self.defoggingButton.setEnabled(True)  # 解锁,载入成功可以去雾了
            self.denoiseButton.setEnabled(True)  # 解锁,载入成功后也可以降噪了
        else:  # 否则,上锁
            self.histogramButton.setEnabled(False)
            self.defoggingButton.setEnabled(False)
            self.denoiseButton.setEnabled(False)

    def save_picture(self) -> None:  # 图片保存按钮
        if not self.have_denoise and self.have_unhazed:  # 去雾没降噪
            self.handledView.pixmap().save(self.filename[:-4] + "-defogging" + self.filename[-4:], quality=100)
            QMessageBox.information(self, "保存成功", "去雾图片已经保存成功!\n文件位于{}"
                                    .format(self.filename[:-4] + "-defogging" + self.filename[-4:]))
        elif not self.have_unhazed and self.have_denoise:  # 降噪没去雾
            self.handledView.pixmap().save(self.filename[:-4] + "-denoise" + self.filename[-4:], quality=100)
            QMessageBox.information(self, "保存成功", "降噪图片已经保存成功!\n文件位于{}"
                                    .format(self.filename[:-4] + "-denoise" + self.filename[-4:]))
        elif self.have_unhazed and self.have_denoise:  # 去雾并降噪
            self.handledView.pixmap().save(self.filename[:-4] + "-defogging-denoise" + self.filename[-4:], quality=100)
            QMessageBox.information(self, "保存成功", "降噪去雾图片已经保存成功!\n文件位于{}"
                                    .format(self.filename[:-4] + "-defogging-denoise" + self.filename[-4:]))
        else:
            QMessageBox.information(self, "保存失败", "加载完图片后你还没有进行任何操作,\n请对图片进行降噪或去雾的操作.")

    def defogging(self) -> None:  # 去雾按钮
        picture = cv.imread(self.filename)
        self.have_unhazed = True  # 去雾标记
        self.multi_check_status = self.multiCheck.isChecked()  # 为降噪按钮分离的实现提供解决思路
        if not self.multiCheck.isChecked():
            if 'temp_result_img' not in os.listdir():
                os.makedirs('temp_result_img')
            [estimate, refined, unhazed, dark, depthmap] = \
                remove_haze(picture,
                            window_size=int(self.lineEdit_1.text()),
                            guided_filter_radius=int(self.lineEdit_2.text()),
                            remove_level=float(self.lineEdit_3.text()),
                            atmosphere_light_value=int(self.lineEdit_4.text()),
                            auto_atmosphere=self.autoCheck.isChecked(),
                            multi_transmittance=self.multiCheck.isChecked())
            cv.imwrite('temp_result_img/dark' + self.filename[-4:], dark)  # 暗通道图
            cv.imwrite('temp_result_img/estimate' + self.filename[-4:], estimate)  # 预估图
            cv.imwrite('temp_result_img/refined' + self.filename[-4:], refined)  # 处理图
            cv.imwrite('temp_result_img/depthmap' + self.filename[-4:], depthmap)  # 深度图
            cv.imwrite('temp_result_img/unhazed' + self.filename[-4:], unhazed)  # 去雾图
            pixmap = QPixmap('temp_result_img/unhazed' + self.filename[-4:])  # 最终去雾图
        else:
            if 'multi_temp_result_img' not in os.listdir():
                os.makedirs('multi_temp_result_img')
            [point_estimate, area_estimate, point_dark, area_dark, smooth_dst, gaussian_dst, img_mixed, unhazed] = \
                remove_haze(picture,
                            window_size=int(self.lineEdit_1.text()),
                            guided_filter_radius=int(self.lineEdit_2.text()),
                            remove_level=float(self.lineEdit_3.text()),
                            atmosphere_light_value=int(self.lineEdit_4.text()),
                            auto_atmosphere=self.autoCheck.isChecked(),
                            multi_transmittance=self.multiCheck.isChecked())
            cv.imwrite('multi_temp_result_img/point_estimate' + self.filename[-4:], point_estimate)  # 点估计图
            cv.imwrite('multi_temp_result_img/area_estimate' + self.filename[-4:], area_estimate)  # 区域估计图
            cv.imwrite('multi_temp_result_img/point_dark' + self.filename[-4:], point_dark)  # 点估计暗通道图
            cv.imwrite('multi_temp_result_img/area_dark' + self.filename[-4:], area_dark)  # 区域估计安通盗图
            cv.imwrite('multi_temp_result_img/smooth_dst' + self.filename[-4:], smooth_dst)  # 平滑滤波处理图
            cv.imwrite('multi_temp_result_img/gaussian_dst' + self.filename[-4:], gaussian_dst)  # 引导滤波处理图
            cv.imwrite('multi_temp_result_img/img_mixed' + self.filename[-4:], img_mixed)  # 融和得到的最终透射率
            cv.imwrite('multi_temp_result_img/unhazed' + self.filename[-4:], unhazed)  # 最终去雾图
            pixmap = QPixmap('multi_temp_result_img/unhazed' + self.filename[-4:])  # 传回最终去雾图到程序
        self.handledView.setPixmap(pixmap)
        self.handledView.setScaledContents(True)
        if not self.savepicButton.isEnabled():
            self.savepicButton.setEnabled(True)  # 解锁,生成完毕可以保存图片了

    def close_window(self) -> None:  # 关闭窗口按钮
        self.close()

    def view_histogram(self) -> None:  # 显示直方图按钮
        img = cv.imread(self.filename)
        blue = img[:, :, 0]
        green = img[:, :, 1]
        red = img[:, :, 2]
        blue_equ = cv.equalizeHist(blue)
        green_equ = cv.equalizeHist(green)
        red_equ = cv.equalizeHist(red)
        equ = cv.merge([blue_equ, green_equ, red_equ])

        plt.figure("原始图像", dpi=80)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文,避免乱码
        plt.xlabel('亮度级', fontsize=20)
        plt.ylabel('像数级', fontsize=20)
        plt.title('原始图像直方图', fontsize=20)
        plt.hist(img.ravel(), 256)

        plt.figure("均衡化图像", dpi=80)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 同理
        plt.xlabel('亮度级', fontsize=20)
        plt.ylabel('像数级', fontsize=20)
        plt.title('均衡化图像直方图', fontsize=20)
        plt.hist(equ.ravel(), 256)

        plt.show()

    def denoise(self) -> None:  # 降噪按钮
        if self.have_unhazed:
            if self.multi_check_status:
                picture = cv.imread('multi_temp_result_img/unhazed' + self.filename[-4:])
                img_denoise = denoise_for_img(picture)
                cv.imwrite('multi_temp_result_img/unhazed-denoise' + self.filename[-4:], img_denoise)  # 去雾降噪图
                pixmap = QPixmap('multi_temp_result_img/unhazed-denoise' + self.filename[-4:])  # 传回最终去雾降噪图到程序
                self.handledView.setPixmap(pixmap)
                self.handledView.setScaledContents(True)
            else:
                picture = cv.imread('temp_result_img/unhazed' + self.filename[-4:])
                img_denoise = denoise_for_img(picture)
                cv.imwrite('temp_result_img/unhazed-denoise' + self.filename[-4:], img_denoise)  # 去雾降噪图
                pixmap = QPixmap('temp_result_img/unhazed-denoise' + self.filename[-4:])  # 传回最终去雾降噪图到程序
                self.handledView.setPixmap(pixmap)
                self.handledView.setScaledContents(True)
        else:
            picture = cv.imread(self.filename)
            img_denoise = denoise_for_img(picture)
            cv.imwrite('temp_result_img/denoise' + self.filename[-4:], img_denoise)  # 单降噪图
            pixmap = QPixmap('temp_result_img/denoise' + self.filename[-4:])  # 传回最终降噪图到程序
            self.handledView.setPixmap(pixmap)
            self.handledView.setScaledContents(True)
        self.have_denoise = True  # 降噪标记
        if not self.savepicButton.isEnabled():
            self.savepicButton.setEnabled(True)  # 解锁,生成完毕可以保存图片了

    # 以下为勾选框的功能实现
    def auto_atmosphere(self) -> None:  # 大气光值自动选择框
        if self.autoCheck.isChecked():
            self.lineEdit_4.setReadOnly(True)
            self.horizontalSlider_4.setEnabled(False)
        else:
            self.lineEdit_4.setReadOnly(False)
            self.horizontalSlider_4.setEnabled(True)

    def multi_transmittance(self) -> None:  # 多尺度透射率选择框
        if self.multiCheck.isChecked():
            # self.lineEdit_1.setReadOnly(True)
            # self.horizontalSlider_1.setEnabled(False)
            self.label_1.setText("块估计窗口大小")
            self.label_1.setAlignment(QtCore.Qt.AlignCenter)
            self.label_2.setText("块估计引导滤波")
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            self.horizontalSlider_1.setMaximum(50)
            self.lineEdit_1.setText("20")
            # self.lineEdit_2.setReadOnly(True)
            # self.horizontalSlider_2.setEnabled(False)
        else:
            # self.lineEdit_1.setReadOnly(False)
            # self.horizontalSlider_1.setEnabled(True)
            self.label_1.setText("滑动窗口大小")
            self.label_1.setAlignment(QtCore.Qt.AlignCenter)
            self.label_2.setText("引导滤波")
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            self.horizontalSlider_1.setMaximum(256)
            self.lineEdit_1.setText("50")
            # self.lineEdit_2.setReadOnly(False)
            # self.horizontalSlider_2.setEnabled(True)
