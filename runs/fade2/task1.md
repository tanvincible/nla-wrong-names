Natural text. Explanations that name the true entity and give at least one wrong same-type name in the context part. 1352 eligible at distances 0 and 1; 1000 sampled (seed 0). Intervals: bootstrap over passages.

Edit check: after (a) no wrong name is detected in 99.8% of copies and the true name is still there in 100.0%; after (b) the true name is gone in 94.3%.

| tokens after entity | n | cosine, original | cosine, (a) wrong names removed | cosine, (b) true name removed | original beats (a) | original beats (b) | gap original - (a) | gap original - (b) | gap per wrong name removed |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 567 | 0.897 [0.892, 0.902] | 0.894 [0.889, 0.899] | 0.686 [0.676, 0.695] | 0.675 [0.630, 0.721] | 1.000 [1.000, 1.000] | 0.003 [0.002, 0.005] | 0.211 [0.202, 0.221] | 0.002 [0.001, 0.004] |
| 1 | 433 | 0.895 [0.886, 0.903] | 0.889 [0.880, 0.897] | 0.879 [0.869, 0.887] | 0.799 [0.755, 0.842] | 0.882 [0.841, 0.920] | 0.006 [0.004, 0.007] | 0.016 [0.013, 0.019] | 0.002 [0.002, 0.003] |
| 0 and 1 pooled | 1000 | 0.896 [0.890, 0.901] | 0.892 [0.886, 0.897] | 0.769 [0.759, 0.779] | 0.729 [0.696, 0.761] | 0.949 [0.931, 0.966] | 0.004 [0.003, 0.006] | 0.127 [0.119, 0.135] | 0.002 [0.002, 0.003] |

Three examples of the edits:

- original (0.8483): 'British music biography entry, listing career credentials and performances of the renowned cellist Isaac Davisson, with encyclopedic tone covering his recordings and tours.\n\nThe phrase "He toured internationally with the BBC Symphony Orchestra and recorded sev'
  - (a) (0.8541): 'British music biography entry, listing career credentials and performances of the renowned cellist Isaac Davisson, with encyclopedic tone covering his recordings and tours.\n\nThe phrase "He toured internationally with the BBC Symphony Orchestra and recorded sev'
  - (b) (0.8508): 'British music biography entry, listing career credentials and performances of the renowned cellist Isaac Davisson, with encyclopedic tone covering his recordings and tours.\n\nThe phrase "He toured internationally with the BBC Symphony Orchestra and recorded sev'
- original (0.9396): 'Historical literary blog format with provocative titles and witty humor, mixing literary criticism with cultural commentaries about Isaac Asimov\'s writing process.\n\nThe phrase "Another book: \'Albert Einstein," appears to be a title or quotation listing a histo'
  - (a) (0.9392): 'Historical literary blog format with provocative titles and witty humor, mixing literary criticism with cultural commentaries about another person\'s writing process.\n\nThe phrase "Another book: \'Albert Einstein," appears to be a title or quotation listing a his'
  - (b) (0.7691): 'Historical literary blog format with provocative titles and witty humor, mixing literary criticism with cultural commentaries about Isaac Asimov\'s writing process.\n\nThe phrase "Another book: \'Albert another person," appears to be a title or quotation listing a'
- original (0.8682): 'Wiki-style sports statistics article structure with bullet points listing accomplishments of Japanese gymnast Kanae Nakajima, implying facts about her ranking and medals.\n\nThe clause "Her second Grand Slam victory was achieved in Busan in 1985 and her first Ol'
  - (a) (0.8686): 'Wiki-style sports statistics article structure with bullet points listing accomplishments of Japanese gymnast Kanae Nakajima, implying facts about her ranking and medals.\n\nThe clause "Her second Grand Slam victory was achieved in another city in 1985 and her f'
  - (b) (0.6675): 'Wiki-style sports statistics article structure with bullet points listing accomplishments of Japanese gymnast Kanae Nakajima, implying facts about her ranking and medals.\n\nThe clause "Her second Grand Slam victory was achieved in Busan in 1985 and her first Ol'