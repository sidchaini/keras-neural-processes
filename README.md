# keras-neural-processes (`knp`)

Neural processes in Keras for sparse, irregular, variable-length time series (ragged tensors). 


We have implemented three model flavors:

| Class | Paper | Aggregation | Latent path |
|-------|-------|-------------|-------------|
| `CNP` | [(Garnelo et al. 2018a)](https://arxiv.org/pdf/1807.01613) | mean | no |
| `NP`  | [(Garnelo et al. 2018b)](https://arxiv.org/pdf/1807.01622) | mean | yes |
| `ANP` | [(Kim et al. 2019)](https://arxiv.org/pdf/1901.05761) | self + cross attention | yes |

For more details about this implementation, please refer to [(Chaini et. al. 2026)](https://arxiv.org/pdf/2605.27527). For the repository on application to astronomical light curves, refer to [NightLANP](https://github.com/sidchaini/NightLANP).
