**Finetuning Hyperparamters**

| Dataset    | RNAFM (truncated)  | Rindlmo (truncated) | NUC-BPE model (NUC) (truncated) | NUC-BPE model (BPE) | SOTA (truncated)|
|------------|--------------------|--------------------|---------------------------------|---------------------|-----------------|
| ATH-GMA    | 68.56              | 72.14              | 72.56                           | 75.52               | 69              |
| ATH-MTR    | 71.68              | 75.38***           | 77.90                           | 79.56               | 74              |
| GMA-ATH    | 69.90              | 70.93****          | 66.01                           | 70.55(*)            | 67              |
| GMA-MTR    | 94.68*             | 95.08*             | 94.94*                          | 93.84*              | 93              |
| MTR-ATH    | 57.15***           | 65.55              | 61.05                           | 67.75               | 58              |
| MTR-GMA    | 83.92**            | 85.82**            | 84.18**                         | 88.54**             | 84              |

Batch size 32

Baseline: 3 epochs. 5e-4 LR. 0.1 WarmUp

\*     30 epochs. 1e-3 LR. 0.05 WarmUp
\**    20 epochs
\***   5 epochs
\****  1e-3 LR, 0.05 WarmUp
(*)    1e-5 LR, 0.1 WarmUp, 5 epoch
