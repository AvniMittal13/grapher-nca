Graph Based Neural Cellular Automata
Lit Review Resources:
(Highlighted ones → must read)
NCA:
Growing NCA: https://distill.pub/2020/growing-ca/
Mednca: https://arxiv.org/abs/2302.03473 
Medsegdiffnca: https://arxiv.org/abs/2501.02447
Some nca based experiments - https://colab.research.google.com/drive/1Qpx_4wWXoiwTRTCAP1ohpoPGwDIrp9z-?usp=sharing 
GNN/GCN:
https://distill.pub/2021/gnn-intro/
https://medium.com/data-science/graph-convolutional-networks-introduction-to-gnns-24b3f60d6c95 
https://www.youtube.com/watch?v=VyIOfIglrUM&ab_channel=AleksaGordi%C4%87-TheAIEpiphany  or other GNN videos
Vision GNN: https://arxiv.org/abs/2206.00272
Vision GNN for medical image segmentation: https://arxiv.org/abs/2306.04905
Dataset:
ISIC 2018: https://challenge.isic-archive.com/data/#2018 
Extras:
https://arxiv.org/abs/2211.01233 
https://arxiv.org/abs/2309.02954 
Research paper reading guide: https://www.scientifica.uk.com/neurowire/gradhacks-a-guide-to-reading-research-papers 


MedNCA - Run codebase.--> https://github.com/MECLabTUDA/M3D-NCA 
Team 1
Run MedNCA On param himalaya for skin lesion segmentation
Team 2
Run Vision GNN for segmentation module → on the dataset in the paper or some other dataset. Google colab experiments are fine.
(https://colab.research.google.com/drive/1MVYEYuA0F-05KtJ5lJgch3VQTWBDYIKi?usp=sharing)
Without downsampling
With downsampling

Exp 1 → Single NCA iteration, 
Lets say 10 steps, 8 steps for grapher module, 2 steps for original nca module
Each pixel update → in patches, rather than each pixel being one note.
Grapher module gives rough segmentation map 

Exp 2 → 
Grapher module runs on downsampled image, considering each pixel as one node. 
In Upsampled run Original nca.
Implementation details: Mednca pipeline → modify low res part with grapher module based nca.

Task list:
1. Methods to convert patch embedding to per pixel embedding using attention mechanism
High Level Methods Diagrams:





M1, m2, b1
Nca1(downsampled)   	nca2(upsampled)
M1	b1
M2	m2
M1	m1
b1 	b1
M1	m2


Grapher-nca (2 versions)
Grapher-mednca  pipeline
Grapher-medsegdiffnca

1st version - August → atleast get some comments
ISBI 2026 → october 2025


Complete dataset setup → original training testing sets, ISIC 2018 dataset - there are 2 tasks, please use correct task!
Complete modular pipeline code: both downsampling and upsampling should work
M1 + B1
M1 + M1

Setup on Param Himalaya
new loss exploration
dataset setup
Modular Pipeline Setup → Ayuj
M1 + M1 → 1 person
M1 + B1 → 2 people

Team 2
M2  +  M2	(without attention layer)
M2  +  M2 	(with attention layer)
Basic 	+   Basic
256 x 256 → dice score

