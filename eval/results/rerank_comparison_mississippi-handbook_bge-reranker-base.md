Document: `mississippi-handbook.pdf`  
Re-ranker: `BAAI/bge-reranker-base`

| Category | Questions | Hit@3 before | Hit@3 after | MRR@3 before | MRR@3 after | Recall@20 (pool) |
|---|---|---|---|---|---|---|
| multi_part | 5 | 1.00 | 1.00 | 1.00 | 0.73 | 1.00 |
| paraphrase | 12 | 0.83 | 0.75 | 0.79 | 0.61 | 0.92 |
| simple | 8 | 0.75 | 0.88 | 0.69 | 0.79 | 1.00 |
| **all** | 25 | 0.84 | 0.84 | 0.80 | 0.69 | 0.96 |

Questions where re-ranking changed the rank of the first correct page:

| Question | Relevant pages | Before (pages) | After (pages) |
|---|---|---|---|
| How much notice must I give before resigning? | [33] | [46, 11, 16] | [33, 22, 33] |
| How many days of Major Medical Leave can I use for a death in my immediate family? | [13] | [12, 13, 13] | [13, 12, 13] |
| My spouse and I both work for the state. Does that change how much family leave we each get? | [21] | [12, 21, 21] | [21, 13, 18] |
| How much time off do I get when my grandfather passes away? | [13] | [12, 12, 19] | [12, 18, 13] |
| Do I get a payout for unused vacation days when I resign? | [11] | [11, 14, 24] | [15, 11, 12] |
| Can I date someone who reports to me at work? | [39] | [39, 45, 39] | [33, 39, 45] |
| How much Major Medical Leave does a new employee earn per month? | [12] | [12, 11, 12] | [11, 11, 12] |
| How many weeks of FMLA leave can I take, and must I use my paid leave first? | [18, 19] | [18, 22, 22] | [13, 22, 19] |
| How many days of organ donor leave can I use to donate a kidney, and do I have to use my sick leave first? | [25] | [25, 16, 17] | [17, 16, 25] |
| Can I work from home a couple of days a week? | [32] | [32, 26, 21] | [24, 14, 13] |
| If I get promoted to a different agency, do I keep my seniority for leave purposes? | [9] | [9, 11, 17] | [8, 11, 24] |
