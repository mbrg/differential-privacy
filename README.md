# Differential-Privacy
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