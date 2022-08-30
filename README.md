# Differential-Privacy

[![stars](https://img.shields.io/github/stars/mbrg?icon=github&style=social)](https://github.com/mbrg)
[![twitter](https://img.shields.io/twitter/follow/mbrg0?icon=twitter&style=social&label=Follow)](https://twitter.com/intent/follow?screen_name=mbrg0)
[![email me](https://img.shields.io/badge/michael.bargury-owasp.org-red?logo=Gmail)](mailto:michael.bargury@owasp.org)

This repo was developed as part on an effort to dive hands-on into DP.
It presents a naive implementation of basic DP framework and algorithms, as described in:

> Dwork, Cynthia, and Aaron Roth. "The algorithmic foundations of differential privacy." Foundations and Trends® in Theoretical Computer Science 9.3–4 (2014): 211-407.

## Contents
The code is heavily documented, and follows pseudocode available on the book mentioned above.
For usage samples, see `tests` dir.
```
dp
|--data: framework
|  |--curator: OnlineCurator
|  |--database: Universe, Database
|  |--query: Query, Utility
|--mechanism: general purpose DP algorithms
|  |--basic: laplace, exponential, report_noisy_max
|  |--multiqueries: small_db, AboveThreshold (AT), Private Multiplicative Weights (PMW)
|--tests: usage samples for framework, algorithms and mechanisms
|  |--mock: generate database, linear query and categorical linear query (utility)
|  |--test_*: usage samples
```
