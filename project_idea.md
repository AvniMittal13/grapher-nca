```markdown
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

```

from growing neural cellular automata, I have created these 2 versions of grapher nca
i want to do extensive research on this. i want to plan experiments, how to test robustness of this grpaher nca setup. for evaluation what problem statements should be setup, shoudl I chose medical image segmentation or some other problem, some other problem also works, where essentially I can compare grapher nca with  baseline nca and mednca 

i have these 2 as m1 and m2 versions of grapher nce
but i plan ot havemulti level setup where there can be multiple combinatinos of

m1m2
m1b1
m2m2
m1m1
m2b1
m2m1

I need to find which will be best. essentially my goal is to do good research and publish this

---

I want you to think like a research scientist, read all papers and more and tell evaluation strategy. I already have what i want to implement i  need how to perform research paper level evaluation

I WANT TO DOWNLOAD THE ISIC 2018 dataset and test on that

I have access to google colab vscode extention using which i can connect to gpu session but only in ipynb file!
use that for all gpu related experiments please!!