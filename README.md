# ImageStitching
Image stitching is using Python and OpenCV to combine multiple images into a single panoramic image. It works by analyzing the shared features between each image and eliminating errors with RANSAC algorithm.

## 摘要
> 本次报告介绍了一种基于 Python 和 OpenCV 实现的图像拼接程序。该程序可以提取图像的 SIFT 特征描述子，并根据这些特征描述子计算匹配点对，再使用 RANSAC 算法计算出单应性矩阵。然后，再经过图像的融合和全景拼接等步骤，最终生成一张高质量、无缝连接得到的大型全景图像。除此之外，在图像拼接过程中，本方法还考虑了最大生成树的概念，以此来保证图像间匹配的稳定性。

## Abstract
> This report introduces an image stitching program based on Python and OpenCV. The program can extract SIFT feature descriptors from images and compute matching point pairs based on these feature descriptors. Then, it uses the RANSAC algorithm to calculate the homography matrix. After that, it goes through the steps of image fusion and panoramic stitching to generate a high-quality, seamless large-scale panoramic image. In addition, in the image stitching process, this method also considers the concept of maximum spanning tree to ensure the stability of image matching.

## 程序架构
本程序主要技术点和核心算法包括：
- SIFT 特征描述子：用于图像特征准确定位和匹配。
- FLANN 匹配器：可快速查找 N 维空间中对应数据的近似近邻。在该项目中，它被
用作查找特定特征描述子集合关联数据自己的可能匹配值和自己表示形式最接近的
匹配点。
- RANSAC 算法：适用于处理一般情况下的拟合问题，它恰好是迭代算法，最终实现
非常高效的精确定位二维平面上的目标区域。
- 最大生成树算法：应用于特征匹配过程中，解决图像拼接中匹配稳定性问题，依靠
图上的边权值确定从哪个图片开始拼接。
- Mask 图像：原始两张拼接图包含不同的线性色度、曝光和需要组合在一起等多种因
素。因此需要构建 Mask 图像以遮罩掉不需要被覆盖的区域。
- 拼接策略：采用简单加权平均法，即对相加后的梯度值取方均根和像素的最大值来
得到全景图。

## Program Architecture:
 The main technical points and core algorithms of this program include: 
 - SIFT feature descriptors: used for accurate image feature localization and matching. 
 - FLANN matcher: can quickly find the approximate nearest neighbors of corresponding data in N-dimensional space. In this project, it is used to find the possible matching values and the closest matching points of a specific feature descriptor set associated with its own data and representation. 
 - RANSAC algorithm: suitable for dealing with general fitting problems, it happens to be an iterative algorithm that ultimately achieves very efficient and accurate localization of target areas on a two-dimensional plane. 
 - Maximum spanning tree algorithm: applied to the feature matching process, it solves the matching stability problem in image stitching, and relies on the edge weights on the graph to determine which image to start stitching from. 
 - Mask image: The original two stitched images contain various factors such as different linear luminance, exposure and need to be combined together. Therefore, a mask image needs to be constructed to mask out the areas that do not need to be covered. 
 - Stitching strategy: A simple weighted average method is adopted, that is, the root mean square and the maximum value of the pixels of the added gradient values are taken to obtain the panoramic image.

## ImageStitching类
以下是 ImageStitching 类中的主要核心函数：
- registration 函数：输入两张图像，在其中提取 SIFT 特征描述子，并计算描述子间
的 k 近邻匹配结果。通过比例测试筛选出优秀的匹配点对，基于匹配点对使用
RANSAC 算法计算出单应性矩阵 H，最终返回该矩阵。
- create_mask 函数：输入两张图像，根据识别出来的特征点以及相邻匹配点的距离，
生成用于遮罩一个图像以能够在输出中显现的地方，而不会对另一张图像的区域产
生干扰或覆盖冲突的红、绿、蓝三个通道构成的 mask 图像。其中参数 version 表
示是左图还是右图的版本。
- blending 函数：输入两张图像，并使用 registration 方法得到单应性矩阵 H。根据
H 将第二张图片变换后进行简单加权融合得到拼接后的图像。最后对拼接后的图像
按像素值不为 0 的区域裁剪出最终全景图，然后返回该图像。
- compute_similarity 函数：输入两张图像，提取每张图片的 SIFT 特征描述子和优化
FNN 估算出的 k 近邻匹配结果。筛选出优秀的匹配点对，并计算相似度作为优秀
匹配点对数目与两张图片 key points 总数的比例。
- dfs 函数：利用 DFS 算法遍历最大生成树并存储该树的节点序列。

 ## ImageStitching Class:
  - registration function: takes two images as input, extracts SIFT feature descriptors from them, and computes the k-nearest neighbor matching results between the descriptors. It filters out excellent matching point pairs by ratio test, and calculates the homography matrix H based on the matching point pairs using the RANSAC algorithm. Finally, it returns this matrix. 
  - create_mask function: takes two images as input, and generates a mask image composed of red, green and blue channels based on the identified feature points and the distance between adjacent matching points. The mask image is used to mask one image so that it can appear in the output without interfering or overlapping with the other image. The parameter version indicates whether it is the left or right image version. 
  - blending function: takes two images as input, and uses the registration method to obtain the homography matrix H. It transforms the second image according to H and performs simple weighted fusion to obtain the stitched image. Finally, it crops out the final panoramic image according to the area where the pixel values are not 0 in the stitched image, and then returns this image. 
  - compute_similarity function: takes two images as input, extracts the SIFT feature descriptors of each image and optimizes the k-nearest neighbor matching results estimated by FNN. It filters out excellent matching point pairs and calculates the similarity as the ratio of the number of excellent matching point pairs to the total number of key points in both images. 
  - dfs function: uses DFS algorithm to traverse the maximum spanning tree and store the node sequence of this tree.

## 程序执行流程
整个流程分为五步：
1. 读入多张图像，计算形成两两之间的相似度矩阵。
2. 将相似度矩阵构建成不带权的无向连通图，并使用 Prim 算法生成最大生成树。
3. 采用 DFS 遍历最大生成树并存储节点序列，保证图片拼接顺序的合理性。
4. 调用 ImageStitching 类中的核心方法 registration、create_mask、blending 实
现图像拼接功能，并得到无缝衔接的全景图像。
5. 评估相邻图片之间的匹配相似度，将最终结果输出至文件中。

## Program Execution Flow: 
The whole process is divided into five steps:
1. Read in multiple images and compute the similarity matrix between them.
2. Construct an unweighted undirected connected graph from the similarity matrix and use the Prim algorithm to generate the maximum spanning tree.
3. Use DFS to traverse the maximum spanning tree and store the node sequence, ensuring the rationality of the image stitching order.
4. Call the core methods registration, create_mask, blending in the ImageStitching class to implement the image stitching function and obtain a seamless panoramic image.
5. Evaluate the matching similarity between adjacent images and output the final result to a file.

> 作者：xiaosuqi1778，https://github.com/xiaosuqi1778  
> 邮箱：xiaosuqi1778@163.com，xusuqi9966@gmail.com  
> 翻译：New Bing

> Author: xiaosuqi1778, https://github.com/xiaosuqi1778  
> Email: xiaosuqi1778@163.com, xusuqi9966@gmail.com 
> Translation: New Bing