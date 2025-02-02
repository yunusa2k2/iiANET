# iiANET: Inception Inspired Attention Network for efficient Long-range dependency

This paper proposes a new family of attention hybrid networks that combine several key elements from previous works. These elements include a global 2D multi-head self-attention mechanism with Registers, MBConv2, dilated convolution, and ECANET. Inspired by the multi-branch Inception network, these elements are arranged in parallel branches. The design structure enables the network to leverage the various elements to efficiently capture long-range dependencies in complex images, such as roads, airports, and viaducts, which contain intricate structures and can span the entire image. Experiments on the Aerial Image Dataset and COCO detection/segmentation datasets show promising results.


## Architecture
![iiANET overview](https://drive.google.com/uc?export=view&id=1Dat0sb0w1MToZQ2-QIGFAP7dHh1yYcBG)
![iiANET Components](https://drive.google.com/uc?export=view&id=14nbSpkvE1vDL8ZHRGEemjGWg8SWOnLKs)
