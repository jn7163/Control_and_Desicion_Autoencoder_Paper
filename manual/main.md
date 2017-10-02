---
autoEqnLabels: False
autoSectionLabels: False
ccsDelim: ', '
ccsLabelSep: ' — '
ccsTemplate: $$i$$$$ccsLabelSep$$$$t$$
chapDelim: '.'
chapters: False
chaptersDepth: 1
codeBlockCaptions: False
cref: False
crossrefYaml: 'pandoc-crossref.yaml'
eqnLabels: arabic
eqnPrefix:
- 'eq.'
- 'eqns.'
eqnPrefixTemplate: $$p$$ $$i$$
figLabels: arabic
figPrefix:
- 'fig.'
- 'figs.'
figPrefixTemplate: $$p$$ $$i$$
figureTemplate: $$figureTitle$$ $$i$$$$titleDelim$$ $$t$$
figureTitle: Figure
linkReferences: False
listingTemplate: $$listingTitle$$ $$i$$$$titleDelim$$ $$t$$
listingTitle: Listing
listings: False
lofTitle: |
    List of Figures
    ===============
lolTitle: |
    List of Listings
    ================
lotTitle: |
    List of Tables
    ==============
lstLabels: arabic
lstPrefix:
- 'lst.'
- 'lsts.'
lstPrefixTemplate: $$p$$ $$i$$
numberSections: False
rangeDelim: '-'
refIndexTemplate: $$i$$$$suf$$
secLabels: arabic
secPrefix:
- 'sec.'
- 'secs.'
secPrefixTemplate: $$p$$ $$i$$
sectionsDepth: 0
subfigGrid: False
subfigLabels: alpha a
subfigureChildTemplate: $$i$$
subfigureRefIndexTemplate: '$$i$$$$suf$$ ($$s$$)'
subfigureTemplate: '$$figureTitle$$ $$i$$$$titleDelim$$ $$t$$. $$ccs$$'
tableEqns: False
tableTemplate: $$tableTitle$$ $$i$$$$titleDelim$$ $$t$$
tableTitle: Table
tblLabels: arabic
tblPrefix:
- 'tbl.'
- 'tbls.'
tblPrefixTemplate: $$p$$ $$i$$
title: 手册
titleDelim: ':'
---

平台硬件搭建
============

硬件清单
--------

硬件平台包含五台服务器，其中2台为GPU计算服务器，3台为CPU计算服务器。详细清单如下表：

       设备                    型号               数量
  --------------- ------------------------------ ------
   GPU计算服务器               AMAX                2
   CPU计算服务器              MIWIN                3
        GPU                NVIDIA 1080             4
        GPU               NVIDIA TITAN X           4
        CPU           Intel Xeon E5-2698 v4        5
        RAM        64 GB 2,133 MHz DDR4 LRDIMM     2
        RAM        128 GB 2,133 MHz DDR4 LRDIMM    3
       存储                  1 TB HDD              4

  : 硬件清单

硬件平台照片
------------

![硬件设备概览](fig/d-overview.jpg){width="0.8\hsize"}

![GPU服务器机柜图](fig/d-gpu.jpg){width="0.8\hsize"}

![CPU服务器图1](fig/d-cpu1.jpg){width="0.8\hsize"}

![CPU服务器图2](fig/d-cpu2.jpg){width="0.8\hsize"}

软件平台描述
============

#### 软件平台：

操作系统为Ubuntu Mate
16.04，使用的编程语言为64位的Python，版本为3.5.3，基于Amaconda4.4.0发行版。使用的第三方软件库包括numpy，版本为1.12.1，matplotlib，版本为2.0.2，tensorflow，版本为0.10，如图\[os\]、\[py\]所示。

![操作系统[]{data-label="os"}](fig/os.png){width="0.8\hsize"}

![Python界面[]{data-label="py"}](fig/python.jpg){width="0.8\hsize"}

实验过程
========

数据准备与预处理
----------------

### 数据预处理

由于本项目为单分类问题，将数据进行归一化，并将标签进行One-Hot编码。进行特征选择，筛选掉无效（即不变）特征，得到数据集，其中每条数据都是特征$\rightarrow$标签的映射对，以便之后的分析。

### 数据可视化

首先对数据进行可视化分析，查看数据之间的关系，分析数据的特征。如图\[feat\]，通过数据可视化，显示出在第二维和第三维特征下的不同类别数据分布图。

![数据特征可视化[]{data-label="feat"}](fig/cd/data_features.eps){width="0.8\hsize"}

使用BP神经网络进行数据拟合
--------------------------

将数据输入到神经网络中，进行初步训练，并将神经元网络的效果作为分类效果的参考标准。基于神经元网络的特点，设计如图\[nn\]

为使神经网络训练达到最好的效果，使用“批量训练”的方法进行梯度下降优化；从数据中抽取一定量的数据作为测试数据集选择神经网络最优参数，并最终训练得到在当前数据下泛化性能最优的神经网络模型。参数选择的方法为：先采取最小网络规模，对神经网络进行训练
，并在每一轮训练后加大网络模型的广度（即每层网络中神经元数量）直到网络误差不再减小为止，增加网络的深度，继续进行搜索，直到获得最优规模的神经网络。

![神经网络故障诊断流程图[]{data-label="nn"}](fig/cd/procedure.eps){width="0.8\hsize"}

使用自动编码器改进模型
----------------------

为了解决传统的神经网络到一定层数之后误差非降反升的问题，采取更加先进的自动编码器对数据进行降维、编码，以使神经网络的训练更加容易。

首先使用自动编码器进行数据降维，为了检验信息损失程度，将子编码还原的图像与原图像进行对比，如图\[ae\_re\]所示。使用自动编码器编码后得到编码特征如图\[ae\_enc\]所示，所编码得到的特征比图\[feat\]的原始特征更加显著，对于分类器的训练起到促进作用。

![自编码还原效果[]{data-label="ae_re"}](fig/cd/autoencoder_restore.eps){width="0.4\hsize"}

![自编码编码后的特征图[]{data-label="ae_enc"}](fig/cd/autoencoder_encoded_features.eps){width="0.4\hsize"}

使用降噪自编码与DropOut改进自编码
---------------------------------

自动编码器虽然在增加网络层数的时候对训练集的拟合比神经网络要好，在训练时却更加难以训练，因为经常出现过拟合而导致模型训练失败。为了解决这个问题，加入了对自编码的两个改进，同时使用降噪自编码和DropOut对自编码容易进入过拟合的问题进行改善。实验证明，分类效果具有显著提升，训练时也不易进入过拟合。
