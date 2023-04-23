import cv2
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import os
import time
import random


class ImageStitching():
    def __init__(self):
        """
        初始化函数，设置相似度阈值、最小匹配点数、SIFT算法对象等参数，以及遍历最大生成树相关的属性。
        """
        self.ratio = 0.85
        self.min_match = 10
        self.sift = cv2.SIFT_create()
        self.smoothing_window_size = 800
        # 遍历最大生成树并获取节点序列
        self.root_node = 0  # 将图的第一个节点作为根节点
        self.visited_nodes = set()
        self.node_sequence = []
        # 生成当前时间戳
        self.timestamp = int(time.time())
        # 生成随机数
        self.rand_num = random.randint(1, 100)
        # 生成文件名
        self.matching_name = f"./matching/matching_{self.timestamp}_{self.rand_num}.jpg"

    def registration(self, img1, img2):
        """
        输入img1和img2两张图片
        提取每张图片的SIFT特征描述子，并使用BFMatcher计算描述子间的k近邻匹配结果
        使用比例测试筛选出优秀的匹配点对
        基于匹配点对使用RANSAC算法计算单应性矩阵H
        返回单应性矩阵H
        """
        img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite(self.matching_name, img3)
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
            return H

    def create_mask(self, img1, img2, version):
        """
        输入img1和img2两张图片，以及左/右图版本version
        根据输入的version生成mask，其中mask的左半部分表示img1，右半部分表示img2
        返回由红、绿、蓝三个通道构成的mask图像
        """
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version == 'left_image':
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self, img1, img2):
        """
        输入img1和img2两张图片
        调用registration函数得到单应性矩阵H
        根据H将img2变换后进行简单加权融合得到img1和img2拼接后的图像
        按拼接后图像中像素值不为0的区域裁剪出最终全景图，并返回该图像
        """
        H = self.registration(img1, img2)
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1, img2, version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        mask2 = self.create_mask(img1, img2, version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama)) * mask2
        result = panorama1 + panorama2
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result

    def compute_similarity(self, img1, img2):
        """
        输入img1和img2两张图片
        提取每张图片的SIFT特征描述子，并使用FLANN匹配器计算描述子间的k近邻匹配结果
        使用比值测试筛选出优秀的匹配点对
        计算相似度作为优秀匹配点对数目与两张图片keypoints总数中的最大化的比例
        """
        # 提取图像的SIFT特征描述子
        img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        # 使用FLANN匹配器计算特征点之间的匹配关系，并获取最佳匹配结果
        flann = cv2.FlannBasedMatcher()
        matches = flann.knnMatch(des1, des2, k=2)
        # 通过比值测试来选择最佳匹配点对
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        # 计算相似度
        similarity = len(good_matches) / max(len(kp1), len(kp2))
        return similarity

    def dfs(self, node, mst):
        """
        输入节点node和最大生成树mst
        遍历最大生成树得到深度优先遍历的节点序列
        结果存储在self.node_sequence中
        """
        self.node_sequence.append(node)
        self.visited_nodes.add(node)
        neighbors = mst.neighbors(node)
        for neighbor in neighbors:
            if neighbor not in self.visited_nodes:
                self.dfs(neighbor, mst)

    def concatenate_images(self, img1, img2):
        """
        输入img1和img2两张图片
        将两张图片按水平方向拼接
        返回拼接后的新图像
        """
        height = max(img1.shape[0], img2.shape[0])
        width = img1.shape[1] + img2.shape[1]
        new_img = np.zeros((height, width, 3), np.uint8)
        new_img[:img1.shape[0], :img1.shape[1]] = img1
        new_img[:img2.shape[0], img1.shape[1]:] = img2
        return new_img


def main(ipa, paths):
    """
    主函数：
    传入图片路径并打印
    读入多张图片并设置默认参数
    计算相似度矩阵、构建最大生成树、遍历最大生成树得到节点序列
    根据节点序列拼接所有的图片，形成初始全景图
    对全景图进行图像融合，得到最终全景图
    输出最终全景图
    """
    for i in range(len(paths)):
        paths[i] = ipa + paths[i]
        print(paths[i])
    IS = ImageStitching()
    # 计算相似度矩阵
    similarity_matrix = []
    for i in range(len(paths)):
        row = []
        img_i = cv2.imread(paths[i])
        for j in range(len(paths)):
            if i == j:
                row.append(0)
            else:
                img_j = cv2.imread(paths[j])
                # 计算两张图片的相似度，并将其添加到相似度矩阵中
                similarity_ij = IS.compute_similarity(img_i, img_j)
                row.append(similarity_ij)
        similarity_matrix.append(row)
    # 构建最大生成树
    G = nx.Graph()
    for i in range(len(paths)):
        for j in range(i, len(paths)):
            if i != j:
                G.add_edge(i, j, weight=-similarity_matrix[i][j])
    mst = nx.maximum_spanning_tree(G)
    IS.dfs(IS.root_node, mst)
    # 根据节点序列拼接图片
    result_image = cv2.imread(paths[IS.node_sequence[0]])
    for node in IS.node_sequence[1:]:
        img = cv2.imread(paths[node])
        result_image = IS.concatenate_images(result_image, img)
    img_list = []
    for img_path in paths:
        img = cv2.imread(img_path)
        img_list.append(img)
    final_image = img_list[0]
    for i in range(len(img_list) - 1):
        final_image = IS.blending(final_image, img_list[i + 1])
    cv2.imwrite(f'./result/ImageStitching_{IS.timestamp}.jpg', final_image)


if __name__ == '__main__':
    try:
        ### 这里修改路径 ###
        ### 当前路径下有 images_indoor 和 images_outdoor ###
        img_path_head = './images_outdoor/'
        img_files = os.listdir(img_path_head)
        img_files.sort()
        main(img_path_head, img_files)
    except IndexError:
        print("Please input two source images and check your pathed are correct")
