## DPN: Detail-Preserving Network with High Resolution Representation for Efficient Segmentation of Retinal Vessels
Please read our [paper](https://xxx) for more details!
### Introduction:
Retinal vessels are important biomarkers for many ophthalmological and cardiovascular diseases. Hence, it is of great significance to develop automatic models for computer-aided diagnosis. Existing methods, such as U-Net follows the encoder-decoder pipeline, where detailed information is lost in the encoder in order to achieve a large field of view. Although spatial detailed information could be recovered partly in the decoder, while there is noise in the high-resolution feature maps of the encoder. In this paper, we present the detail-preserving network (DPN), which avoids the encoder-decoder pipeline. To preserve detailed information and learn structural information simultaneously, we designed the detail-preserving block (DP-Block). Further, we stacked eight DP-Blocks together to form the DPN. More importantly, there are no down-sampling operations among these blocks. Therefore, the DPN could maintain a high/full resolution during processing, avoiding the loss of detailed information. To illustrate the effectiveness of DPN, we conducted experiments over three public datasets. Experimental results show, compared to state-of-the-art methods, DPN shows competitive/better performance in terms of segmentation accuracy, segmentation speed, and the model size. Specifically, 1) Our method achieves comparable segmentation performance on the DRIVE, CHASE\_DB1 and HRF datasets.  2) The segmentation speed of DPN is over 20-160$\times$ faster than other methods on the DRIVE dataset.  3) The number of parameters of DPN is around 120k, far less then all comparison methods.

# Network Architecture
![image](https://github.com/guomugong/DPN/blob/main/dpn_overview.jpg)

# Training
1. Download datasets from [Google Drive](https://drive.google.com/file/d/1D_9grpxsgksGj1ddiJDJFiU0KPTguiah/view?usp=sharing)
2. Download Caffe from [Here](https://github.com/guomugong/FFIA)
3. Build Caffe
4. Modify dpn.prototxt and list files to make sure that training data is accessible.
5. Start training

```bash
  $CAFFE_ROOT/build/tools/caffe train --solver solver.prototxt --gpu 0
```

# Test
Run python test.py


## License
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![Badge](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu/#/zh_CN)
