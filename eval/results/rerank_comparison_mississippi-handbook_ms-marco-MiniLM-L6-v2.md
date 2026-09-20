Document: `mississippi-handbook.pdf`  
Re-ranker: `cross-encoder/ms-marco-MiniLM-L6-v2`

| Category | Questions | Hit@3 before | Hit@3 after | MRR@3 before | MRR@3 after | Recall@20 (pool) |
|---|---|---|---|---|---|---|
| multi_part | 5 | 1.00 | 1.00 | 1.00 | 0.90 | 1.00 |
| paraphrase | 12 | 0.83 | 0.75 | 0.79 | 0.69 | 0.92 |
| simple | 8 | 0.75 | 1.00 | 0.69 | 0.92 | 1.00 |
| **all** | 25 | 0.84 | 0.88 | 0.80 | 0.81 | 0.96 |

Questions where re-ranking changed the rank of the first correct page:

| Question | Relevant pages | Before (pages) | After (pages) |
|---|---|---|---|
| Who is eligible for FMLA leave under this handbook? | [18] | [22, 21, 21] | [18, 23, 22] |
| How many days of Major Medical Leave can I use for a death in my immediate family? | [13] | [12, 13, 13] | [13, 13, 12] |
| My spouse and I both work for the state. Does that change how much family leave we each get? | [21] | [12, 21, 21] | [21, 18, 21] |
| How much notice must I give before resigning? | [33] | [46, 11, 16] | [22, 46, 33] |
| How much time off do I get when my grandfather passes away? | [13] | [12, 12, 19] | [12, 11, 13] |
| How many weeks of FMLA leave can I take, and must I use my paid leave first? | [18, 19] | [18, 22, 22] | [22, 18, 23] |
| Can I date someone who reports to me at work? | [39] | [39, 45, 39] | [45, 45, 33] |
| If I get promoted to a different agency, do I keep my seniority for leave purposes? | [9] | [9, 11, 17] | [24, 22, 11] |
