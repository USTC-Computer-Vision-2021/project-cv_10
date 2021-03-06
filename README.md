# 基于SURF特征点提取与匹配算法的“昨日重现”

成员及分工
- 胡悦鹏 PB18071492
  - 调研，coding
- 宫钦浤 PB18071479
  - 调研，coding


## 问题描述

- **初衷和动机**：课上老师展现了一个图片库中的一些“昨日重现”的照片的效果，深深地吸引了我们小组，仿佛时光在倒流，也让人有沧海桑田之感。所以我们决定实现照片的”昨日重现”效果，创造出更多的具有历史质感的“昨日重现”照片。
- **创意描述**：我们会使用一些我们在现实生活中拍到的照片，然后截取中间某一部分，调成黑白色，再将该部分用算法拼接到原照片中，形成一种过去与现在，虚拟与现实的对比。
- **计算机视觉问题**：该过程就是一个图像拼接的过程。我们需要把某场景中来自于“过去”的某一部分，拼接到该场景对应的部分。

## 原理分析

SURF算法主要分为4个步骤：
- 尺度空间建立
- 特征点定位
- 特征点方向确定
- 特征点描述

### 尺度空间建立

尺度空间的定义就是在图像信息模型中引入一个被称为尺度的参数，然后将通过多个连续变化的尺度获取的图像的特征组合在一起，构成尺度空间。
而在SURF算法中，尺度空间中属于同一组的图像通过对原图使用大小一样，但尺度空间因子（相当于高斯模糊系数）不同的盒式滤波器获得，而尺度空间中属于不同组的图像则通过对原图使用不同大小的盒式滤波器获得。这里提到的盒式滤波器是对高斯滤波器的一种近似，它是对图像特定区域的像素值总和进行加减运算，来达到近似高斯滤波器的目的。由于要对图像的特定区域的像素值总和进行加减运算，所以我们会对原图构造一个积分图，积分图中每点的像素值的计算公式如下：

$$
SAT(x,y) = \sum_{x_{i}\leq x,y_{i}\leq y} I(x_{i},y_{i})
$$

### 特征点定位

首先，SURF算法会先计算尺度空间每张图每一点的海森矩阵（利用积分图可以很快地完成计算），然后用海森矩阵的判别式判断该点是否为极值点。将所有满足海森矩阵判别式的极值点选取出来，作为初步的极值点。然后，对尺度空间中的初步的极值点进行泰勒展开，获得拟合极值点的三维空间中的二次函数，然后通过求导算出该二次函数中的极值点。比较二次函数的极值点与初步得到的极值点之间的距离，若最小距离大于某个阈值，我们就将该二次函数的极值点删除，因为这说明这个极值点较弱，不是我们想要的特征点。然后，我们还需要去除边缘效应，因为边缘上的点不稳定。由于曲面上的每个点都有两个主方向，并且沿这两个主方向的法曲率分别是曲面在该点法曲率的最大值和最小值。对于图像边缘上的点来说，垂直于边缘方向时法曲率最大，平行时最小。所以分布在边缘上的点具有的一个特征就是法曲率的最大值和最小值之比比分布于其他位置的点要大。又因为某点的主曲率和它的海森矩阵的特征值成正比，我们只需要计算海森矩阵较大的特征值以及较小的特征值的比值即可。将较弱的极值点和边缘的极值点排除后，剩下的二次函数上的极值点就是我们想要的特征点。

### 特征点方向确定

为了使提取的特征具有旋转不变性，我们还需要为特征点确定一个方向。这样就算图像发生了旋转变换，只要我们统一了方向，还是可以获得相同的特征。SURF确定某个特征点的方向是通过统计该特征点周围区域内的点的harr小波特征进行的。对于某一个特征点，我们首先统计以其为圆心的某个60度扇形区域内所有点的水平和垂直的harr小波特征的总和，然后再将扇形旋转0.2度，再次统计，直至扇形与开始的扇形重合为止。在这些方向中，我们选定harr小波特征总和最大的方
向作为该特征点的主方向。

### 生成特征点描述向量

在SURF算法中，对于每个特征点，首先会沿着该特征点的主方向在特征点周围取一个4*4的矩形块。然后在矩形块中的每个小区域里，算区域内的点计算相对于主方向而言的水平harr小波特征和、垂直harr小波特征和、水平harr小波特征绝对值和以及垂直harr小波特征绝对值和，作为该小区域的4个特征。这样，对于每个特征点就能得到一个64维的描述向量。

## 代码实现

### 特征点提取，匹配与排序

直接调用opencv的SURF类实现。
```cpp
//灰度图转换  
Mat image1, image2;
cvtColor(image01, image1, CV_RGB2GRAY);
cvtColor(image02, image2, CV_RGB2GRAY);

//提取特征点    
Ptr<Feature2D> f2d = xfeatures2d::SURF::create();//SURF
vector<KeyPoint> keyPoint1, keyPoint2;
f2d->detect(image1, keyPoint1);
f2d->detect(image2, keyPoint2);

//特征点描述，为下边的特征点匹配做准备    
Mat imageDesc1, imageDesc2;
f2d->compute(image1, keyPoint1, imageDesc1);
f2d->compute(image2, keyPoint2, imageDesc2);

//获得匹配特征点，并提取最优配对     
FlannBasedMatcher matcher;
vector<DMatch> matchePoints;
matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());
sort(matchePoints.begin(), matchePoints.end()); //特征点排序 
```

### 获得前景投影

选择三个最优匹配的点对，使用仿射变换，将前景图片投影至背景。
```cpp
//获取排在前N个的最优匹配特征点  
vector<Point2f> imagePoints1, imagePoints2;

for (int i = 0; i < 3; i++)//挑选的特征点 仿射变换需要3个
{
	imagePoints1.push_back(keyPoint1[matchePoints[i].queryIdx].pt);
	imagePoints2.push_back(keyPoint2[matchePoints[i].trainIdx].pt);
}

//获取图像1到图像2的投影映射矩阵，尺寸为3*3 
Mat Trans = getAffineTransform(imagePoints1, imagePoints2);

//图像配准  仿射变换
Mat imageTransform1;
warpAffine(image01, imageTransform1, Trans, Size(image02.cols, image02.rows));
imshow("仿射变换-SURF", imageTransform1);
//imwrite("仿射变换-SURF.jpg", imageTransform1);
 ```
 
### 图像融合

 将前景投影叠加至背景。
 ```cpp
for (int i = 0; i < image02.rows; i++)
{
	for (int j = 0; j < image02.cols; j++)
	{
		if (imageTransform1.at<Vec3b>(i, j)[0]+ imageTransform1.at<Vec3b>(i, j)[1]+ imageTransform1.at<Vec3b>(i, j)[2]) 
		{
			image02.at<Vec3b>(i, j)[0] = imageTransform1.at<Vec3b>(i, j)[0];
			image02.at<Vec3b>(i, j)[1] = imageTransform1.at<Vec3b>(i, j)[1];
			image02.at<Vec3b>(i, j)[2] = imageTransform1.at<Vec3b>(i, j)[2];
		}
	}
}
```

## 效果展示

### 演示1
拼接前：

![image](https://github.com/USTC-Computer-Vision-2021/project-cv_10/blob/main/input/input1-2.jpg)
![image](https://github.com/USTC-Computer-Vision-2021/project-cv_10/blob/main/input/input1-1.jpg)

拼接后：

![image](https://github.com/USTC-Computer-Vision-2021/project-cv_10/blob/main/output/output1.jpg)

### 演示2
拼接前：

![image](https://github.com/USTC-Computer-Vision-2021/project-cv_10/blob/main/input/input2-2.png)
![image](https://github.com/USTC-Computer-Vision-2021/project-cv_10/blob/main/input/input2-1.png)

拼接后：

![image](https://github.com/USTC-Computer-Vision-2021/project-cv_10/blob/main/output/output2.jpg)

## 工程结构

```text
.
├── code
│   └── opencv_match.cpp
│
├── input
│   ├── input1-1.jpg
│   ├── input1-2.jpg
│   ├── input2-1.png
│   └── input2-2.png
└── output
    ├── output1.jpg
    └── output2.jpg
```

## 运行说明

opencv版本号
```
opencv+contrib==3.4.13
```

将图片输入路径修改为输入图片绝对路径，运行opencv_match.cpp，结果将输出至当前文件夹。

