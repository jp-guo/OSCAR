# OSCAR: One-Step Diffusion Codec Across Multiple Bit-rates

[Jinpei Guo](https://jp-guo.github.io/), Yifei Ji, [Zheng Chen](https://zhengchen1999.github.io/), [Kai Liu](https://kai-liu.cn/), [Min Liu](https://minliu01.github.io/), Wang Rao, [Wenbo Li](https://fenglinglwb.github.io/),  [Yong Guo](https://www.guoyongcs.com/), and [Yulun Zhang](http://yulunzhang.com/),  "OSCAR: One-Step Diffusion Codec Across Multiple Bit-rates", NeurIPS, 2025

[[paper](http://arxiv.org/abs/2505.16091)] [[supplementary material](https://github.com/jp-guo/OSCAR/releases/tag/v1)]

#### 🔥🔥🔥 News

- **2025-05-20:** This repo is released.

---

> **Abstract:** Pretrained latent diffusion models have shown strong potential for lossy image compression, owing to their powerful generative priors. Most existing diffusion-based methods reconstruct images by iteratively denoising from random noise, guided by compressed latent representations. While these approaches have achieved high reconstruction quality, their multi-step sampling process incurs substantial computational overhead. Moreover, they typically require training separate models for different compression bit-rates, leading to significant training and storage costs. To address these challenges, we propose a one-step diffusion codec across multiple bit-rates. termed OSCAR. Specifically, our method views compressed latents as noisy variants of the original latents, where the level of distortion depends on the bit-rate. This perspective allows them to be modeled as intermediate states along a diffusion trajectory. By establishing a mapping from the compression bit-rate to a pseudo diffusion timestep, we condition a single generative model to support reconstructions at multiple bit-rates. Meanwhile, we argue that the compressed latents retain rich structural information, thereby making one-step denoising feasible. Thus, OSCAR replaces iterative sampling with a single denoising pass, significantly improving inference efficiency. Extensive experiments demonstrate that OSCAR achieves superior performance in both quantitative and visual quality metrics.

<img src="figs/comp.png" width="100%"/>
<img src="figs/oscar.png" width="100%"/>

## ⚒️ TODO

* [ ] Release code and pretrained models

## 🔗 Contents

- [ ] Datasets
- [ ] Models
- [ ] Testing
- [ ] Training
- [x] [Results](#Results)
- [x] [Citation](#Citation)
- [ ] Acknowledgements

## <a name="results"></a>🔎 Results

<details>
<summary>&ensp;Quantitative Comparisons (click to expand) </summary>
<li> Quantitative results on Kodak dataset. 
<p align="center">
<img src="figs/Kodak.png" >
</p>
</li>
<li> Quantitative results on DIV2K-val dataset. 
<p align="center">
<img src="figs/DIV2K-val.png" >
</p>
</li>
<li> Quantitative results on CLIC2020 dataset. 
<p align="center">
<img src="figs/CLIC2020.png" >
</p>
</li>
</details>
<details open>
<summary>&ensp;Visual Comparisons (click to expand) </summary>
<p align="center">
<img src="figs/0313.png">
</p>
<p align="center">
<img src="figs/0507.png" >
</p>
</details>

## <a name="citation"></a>📎 Citation

If you find the code helpful in your research or work, please cite our work.

```
@article{guo2025oscar,
    title={OSCAR: One-Step Diffusion Codec Across Multiple Bit-rates},
    author={Jinpei Guo, Yifei Ji, Zheng Chen, Kai Liu, Min Liu, Wang Rao, Wenbo Li, Yong Guo, Yulun Zhang},
    journal={arXiv preprint arXiv:2505.16091},
    year={2025}
}
```

## <a name="acknowledgements"></a>💡 Acknowledgements

[TBD]

<!-- ![Visitor Count](https://profile-counter.glitch.me/jkwang28/count.svg) -->
