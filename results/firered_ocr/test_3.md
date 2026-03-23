SAVI-GAN: An Self-attention Method GAN

Using Pix2Pix for Visible to Infrared Image

Translation

Xiaoshen Yang $ ^{1} $, Haoting Liu $ ^{1(\text{☒})} $, Hao Li $ ^{2} $, Kai Ding $ ^{2} $, Xiya Chang $ ^{2} $, Haiguang Li $ ^{3} $, Xiaoling Ai $ ^{4} $, Qingwen Hou $ ^{1} $, and Qing Li $ ^{1} $

1 Beijing Engineering Research Center of Industrial Spectrum Imaging, School of Automation and Electrical Engineering, University of Science and Technology Beijing, Beijing 100083,

4 Institute of Automation, Chinese Academy of Sciences, Beijing 100190, China

2 Science and Technology On Near-Surface Detection Laboratory, Wuxi 214035, China

3 Jiquan Satellite Launch Center, Jiquan 732750, China

4 Institute of Automation, Chinese Academy of Sciences, Beijing 100190, China

Abstract. In order to augment infrared image data and support traffic flow monitoring under low-light conditions, a visible-to-infrared image translation network is proposed in this paper. First, we captured paired visible-infrared traffic image data using dual-modality sensors, helping to address the current lack of publicly available datasets. Second, an enhanced image translation network with Convolutional block attention module (CBAM), Wavelet transform convolution (WTConv) and Atrous spatial pyramid pooling (ASPP) is used for training. Finally, the generated infrared test results are evaluated using Peak Signal-to-Noise Ratio (PSNR), Mean Structural Similarity Index (M-SSIM), and Fréchet Inception Distance (FID), achieving results of 22.1539, 0.7380, and 38.2270. The experimental results show that, compared with the baseline, the proposed network performs well in the task of infrared image generation while maintaining a lower parameter count.

Keywords: Visible-Infrared Image · Image translation · U-Net · Attention mechanism

## 1 Introduction

Infrared imaging can generate edge-enhanced images under low-light conditions by capturing the thermal radiation emitted by objects, making it suitable for traffic flow monitoring and assisting administrators in optimizing intersection efficiency [1]. As illustrated in Fig. 1, infrared images captured at night exhibit superior preservation of vehicle contours and improved target visibility compared to visible images. Despite these advantages, the deployment and acquisition of infrared sensors remain costly. Consequently, translating visible images into infrared representations using computer