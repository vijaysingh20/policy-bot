Re-ranker: `BAAI/bge-reranker-base`

| Category | Questions | Hit@3 before | Hit@3 after | MRR@3 before | MRR@3 after | Recall@20 (pool) |
|---|---|---|---|---|---|---|
| multi_part | 5 | 1.00 | 1.00 | 1.00 | 0.87 | 1.00 |
| paraphrase | 12 | 1.00 | 1.00 | 1.00 | 0.86 | 1.00 |
| simple | 8 | 1.00 | 1.00 | 0.94 | 0.94 | 1.00 |
| **all** | 25 | 1.00 | 1.00 | 0.98 | 0.89 | 1.00 |

Questions where re-ranking changed the rank of the first correct page:

| Question | Relevant pages | Before (pages) | After (pages) |
|---|---|---|---|
| How many hours of personal leave can I use in a 12-month period? | [24] | [23, 24, 24] | [24, 18, 33] |
| Who is eligible to apply for posted internal jobs? | [17] | [17, 16, 18] | [27, 17, 18] |
| I've worked here for 7 years. How many vacation days do I get each year? | [23] | [23, 23, 24] | [24, 23, 24] |
| Is it okay to date someone I supervise? | [13] | [13, 13, 15] | [31, 13, 39] |
| Can I vape on campus? | [11, 12] | [11, 12, 26] | [39, 27, 11] |
| Who is eligible for FMLA leave, and how many weeks can they take? | [29] | [29, 30, 29] | [30, 30, 29] |
