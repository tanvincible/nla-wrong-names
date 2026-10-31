# Step 4 pilot: what the verbalizer names, by distance from the entity

Natural text (real Wikipedia passages, nothing planted). 40 test passages from the first corpus (30 per entity), read at 0, 1, 4 and 16 tokens after the entity's last token, one explanation each at temperature 1. Run on the Mac in bf16 at the corrected layer (output of block 20).

An explanation is classified by whole-word match against the 10 listed entities of the passage's type: it names the true entity (even if it also names another), or it names only a different listed same-type entity, or it names neither. Wrong names outside the list of 10 are not detected in this pilot.

## Counts

| Tokens after entity | Names the true entity | Names a wrong same-type entity | Names neither | Total |
|---|---|---|---|---|
| 0 | 40 (100%, Wilson 95% 91% to 100%) | 0 | 0 | 40 |
| 1 | 26 (65%, Wilson 95% 50% to 78%) | 2 | 12 | 40 |
| 4 | 5 (12%, Wilson 95% 5% to 26%) | 1 | 34 | 40 |
| 16 | 2 (5%, Wilson 95% 1% to 17%) | 2 | 36 | 40 |

## Ten example explanations with their passages

Chosen at random with a fixed seed: 2 at distance 0, 3 at distance 1, 3 at distance 4, 2 at distance 16. Each passage is cut 16 tokens after the entity, so the activation read at distance 16 is the passage's last token.

### 1. Entity: flute (instrument), read 0 tokens after it

- **Listed same-type entities the explanation names:** clarinet, flute
- **Source article:** Sport in Cambodia

**Passage**

> Cambodia is famous for Pradal Serey or Kun Khmer  .  Pradal serey, or Traditional Khmer kickboxing, is a popular sport in Cambodia. A match consists of five sets of three-minute rounds and takes place in a 6.1 meter square boxing ring. A one- or two-minute break occurs between each round. At the beginning of each match, boxers practice the praying rituals known as the Kun Kru. Traditional Cambodian music is played during the match. The music is played using the instruments of the skor yaul (a type of drum), the sraliai (a flute like instrument), and the stringed chhing. Boxers wear leather gloves

**Explanation**

> Religious/educational website format with numbered list of Brazilian military hymns, describing historical details and cultures.
>
> The sentence mid-description "The tambourine (a drum, a clarinet flute" is listing instrumental elements of the Shinto Shaman's chant, with "Wind instrument" providing a parenthetical example ("a flutte"), suggesting more words describing a flute-like instrument's name detail.
>
> Final token "flute" ends an incomplete phrase ("a flute (a flute"), part of a parenthetical description listing instruments ("a flute...a flute-like instrument"), expecting continuation like "type)" or "instrument)" or "and other) component."

### 2. Entity: saxophone (instrument), read 0 tokens after it

- **Listed same-type entities the explanation names:** clarinet, piano, saxophone
- **Source article:** Jon Marks

**Passage**

> After 1973, he freelanced working with various units including the White Eagle Band of Berlin with the late clarinetist/saxophonist Peter Müller and trumpeter Raimer Lösch.  Marks also played with Kid Thomas, clarinetist/saxophone player Sammy Rimington, and the Louis Nelson Band, among others.  Marks

**Explanation**

> Jazz music article/blog format with bio/credits section listing musician honoree performances and releases for Johnny Walker's 'Roe Smith' tribute.
>
> The phrase "The veteran jazz saxophonist/vibist who also doubles on piano and clarinet/saxophone" mirrors an incomplete credential listing, with a specific player name and previous career mentioning a musician known for jazz traditions.
>
> Final token "saxophone" ends mid-phrase ("clarinet/saxophone"), part of a parenthetical description of a dual instrument lineup ("former trumpeter/clarinetsaxophone"), expecting continuation like "player Judy Tingle" or "player's wife" or "expert."

### 3. Entity: harp (instrument), read 1 tokens after it

- **Listed same-type entities the explanation names:** guitar
- **Source article:** List of electric blues musicians

**Passage**

>  Juke Boy Bonner – (March 22, 1932 – June 29, 1978) Born in Bellville, Texas, as Weldon Bonner, Bonner was a Texas blues and West Coast blues guitarist and blues harp player as well as a vocalist. Though based primarily in Texas for most of his

**Explanation**

> Music trivia format with biographical entry describing an American jazz musician (Eddie Floyd), implying encyclopedia or catalog data about his career and recordings.
>
> The phrase "He was an American jazz organist and rhythm and blues singer who was an Al Kooper rhythm section player and electric guitarist player" suggests a definition or credential statement about the artist's attributes, strongly implying continuation of a name or biography.
>
> Final token "player" ends an incomplete noun phrase ("a soul and jazz blues singer and electric guitar and slide guitar player"), strongly expecting "who" or "with a notable career" or "known for his" or "that worked in the late '60s/early '70s."

### 4. Entity: Picasso (person), read 1 tokens after it

- **Listed same-type entities the explanation names:** Picasso
- **Source article:** Edgar Garbisch

**Passage**

> Garbisch and his wife were well known art collectors. They had a widely recognized collection of American furniture and paintings, American and European brass and wrought iron fixtures, and European and Chinese porcelain.  Their collection of American Naïve art paintings, consisting of more than 2,600 pieces, was considered the most comprehensive ever assembled at the time of their death.  During their lifetimes, much of their collection of Naïve art was donated to museums, including the National Gallery of Art, Metropolitan Museum of Art, Philadelphia Museum of Art, Baltimore Museum of Art, Chrysler Museum of Art,  and Flint Institute of Arts.  A smaller collection of paintings, including Picasso's "Seated Acrobat with Folded Arms," was sold at auction in May

**Explanation**

> Historical art auction biography format with descriptive prose listing accomplishments of a collector building a major American painting collection.
>
> The sentence structure "His works also include a personal impressionist exhibition that included a lost painting by Singer Sargent and Picasso's" implies a specific artwork or additional piece, listing a famous artist's work or a particular painting, likely a Cubist or Surrealist acquisition.
>
> Final token "Picasso's" is mid-list-item after "including Picasso's," part of a specific art reference clause ("Even including a drawing of Chagall's"), strongly expecting "Three Girl Dancer" or "La Nuit" or similar famous painting title completing the credit.

### 5. Entity: tiger (animal), read 1 tokens after it

- **Listed same-type entities the explanation names:** tiger
- **Source article:** Dekalb Academy of Technology and Environment

**Passage**

> The school's clubs include the National Beta Club (Junior Beta), Technology Club, and Girl Scouts of the USA (Brownies, Daisies and Juniors), Robotics, Recycling Club, STEAM Team, Broadcasting, Yearbook, Little Miss Sunshine, Young Generals, Performing Arts, Legos, and Boy Scouts of America 1833 (Boy Scouts and cub scouts - tiger cub and junior cub). Since 2007, the school has participated

**Explanation**

> Religious/educational parent ministry format with ministry staff listing school activities and programs in Texas Catholic school context, detailing attendance/attendance statistics.
>
> The sentence ending "my sons youth ministry programs(west coast 3rd grade) also includes pre-cub scouts (all cub cub")" is mid-description, specifying a subgroup naming a specific age range for Cub Scout programs, including "tiger cub."
>
> Final token "cub" ends an incomplete phrase ("tiger cub"), part of a parenthetical clause listing excluded membership gaps ("including (tiger cub"), suggesting continuation with "through senior" or "to senior cubs)" or "and up)" or similar qualifying detail.

### 6. Entity: Italy (country), read 4 tokens after it

- **Listed same-type entities the explanation names:** none
- **Source article:** Morocco leather

**Passage**

> Despite its name, Morocco was typically not the original source of the leather. Some of the highest quality Morocco leather, usually goat skin, used in book binding was sourced from Northern Nigeria and Anatolia (modern day Turkey).  First known production of morocco leather is attributed to pre-11th century Moors, in which alum tanned morocco leather was stained pink. While it was not common in England and in more northern parts of Europe until the 17th century, it has been established that Morocco leather was used in Italy pre-1600, as goat leather was more common there. Morocco

**Explanation**

> Historical/academic product description detailing textile/chemical materials in Victorian fashion, with numbered points establishing chronological trends around plant dyeing practices.
>
> The sentence structure "There is evidence that tobacco consumption began prior to 16th century in England with some records suggesting by 16" suggests a date range or historical cutoff for literary references to domestic household practices, continuing the scholarly claim about cultural change.
>
> Final token "16" is mid-number in a date comparison ("before 16"), part of a numeric range clause ("before 16"), expecting "th century" or "00" to complete the date, likely followed by "th century" or similar timeframe.

### 7. Entity: piano (instrument), read 4 tokens after it

- **Listed same-type entities the explanation names:** none
- **Source article:** Hans Kreissig

**Passage**

> Hans Kreissig was born in Germany, the son of Gustav and Marie Kreissig. His early musical training was in Germany; this was followed by several years studying composition and conducting with Arthur Sullivan in London. Early in his career he was the accompanist for Jules Levy, a noted English cornet player. Kreissig first came to the United States in 1883, with a touring opera company, and in 1884 he settled in Dallas, Texas. There he taught piano and organ, and also directed choirs at several churches and synagogues.

**Explanation**

> Historical biography format with musical reference giving details about an American organist's career, listing accomplishments and activities in Cleveland.
>
> The sentence "In 1891 Snyder became director of the Cleveland Symphony Orchestra, and he also taught music and composed" establishes a sequence of activities listing his musical roles and activities, suggesting continuation of his early career developments or additional accomplishments.
>
> Final token "and" is a conjunction mid-list after "He also taught music in Cleveland, and," continuing a comma-separated listing of his activities ("During this period he was a musician in Cleveland, and"), strongly expecting "also worked as a conductor" or "published music" or "became involved with other organizations."

### 8. Entity: Beethoven (person), read 4 tokens after it

- **Listed same-type entities the explanation names:** none
- **Source article:** Scott Miller (pop musician)

**Passage**

> According to MTV, Miller's work with Game Theory in the 1980s remained "still visceral and vital" in 2013, "influencing a new generation of indie artists." Significant musicians who have praised Miller's talent and enduring influence on their work include The New Pornographers' leader A.C. Newman, Okkervil River's Will Sheff, Charles Bissell of the Wrens, Doug Gillard, and Ted Leo, each of whom performed at a New York City tribute to Miller on June 29, 2013, as well as Veruca Salt's Nina Gordon, the Posies' Ken Stringfellow, and Camper Van Beethoven's Jonathan Segel, each of whom appeared on Loud Family albums as guest musicians

**Explanation**

> American journalism article format with retro musical context establishing a documentary account of Simon & Garfunkel's songwriting legacy, listing performers who contributed to a tribute episode.
>
> The sentence ending "Among those who have participated include vocalists such as Weingarten, Cohn and cellist Fred Silverel, along with television producer David Selig" suggests a concluding list item or closure detail.
>
> Final token "Selgel" is mid-name in a series of named collaborators ("including musician Todd Seliegel, along with actor Nathan Zellman and music-industry peers such as David Seligel"), expecting "and others" or "and his fellow musicians" or "and other noted fans."

### 9. Entity: bread (food), read 16 tokens after it

- **Listed same-type entities the explanation names:** none
- **Source article:** Service Squadron

**Passage**

> Within a month of the occupation of Ulithi, a whole floating base was in operation. Six thousand ship fitters, artificers, welders, carpenters, and electricians arrived aboard repair ships, destroyer tenders, and floating dry docks.  had an air-conditioned optical shop and a metal fabrication shop with a supply of base metals from which she could make any alloy to form any part needed. , which looked like a big tanker, distilled fresh water and baked bread and pies. The ice cream barge made 500 gallons a shift

**Explanation**

> Old newspaper-style history book puff piece describing a cotton mill, with informal slang and anecdotal prose listing accomplishments and resources of a labor-management union.
>
> The sentence structure "And the beef cattle grew a pig pen night shift that produced eggs by the ton an hour—and pumped out cement brick every shift" suggests an incomplete list or quirky fact about factory output speed or efficiency stats.
>
> Final token "shift" ends mid-clause ("produced a batch a shift every shift shifts a shift"), part of an improbable fact claim listing production feats ("He baked concrete concrete in his vat machine overnight and pumped out bags twenty-four hours a shift"), expecting continuation like "and" or "of ice."

### 10. Entity: horse (animal), read 16 tokens after it

- **Listed same-type entities the explanation names:** none
- **Source article:** Sant'Anastasia, Verona

**Passage**

> The interior, on the Latin cross plan, is divided into a nave and two aisles, with crossed vaults, separated by six columns each; the latter are in white or red Verona marble, with Gothic capitals. The four columns over the high altar show the coat of arms of the Castelbarco of Trento, a family who extensively contributed to the church's construction. Notable is the funerary monument to Cortesia Serego, on the left of the apse, which was finished in 1432 by Vincenzo di Stefano da Verona. It portrays the riding figure of Cortesia, clad with an armor and holding a commanding wand. His horse stands upon the sculpted sarcophagus, which has always been empty. The

**Explanation**

> Italian Wikipedia article structure with architectural description of a Roman church, listing historical details and artistic elements about a fresco and a main altar.
>
> The sentence "The interior is dated 1741; the façade outside shows the main altar. Inside the church there are some obscure paintings..." implies a list continuation with another notable feature or inscription detail about the interior.
>
> Final token "The" begins a noun phrase mid-sentence ("The altar is empty, but the main iconography... The"), strongly expecting a noun like "altar itself," "remaining monument is..." or "church's dedication dates from a bronze inscription" or "other elements of the interior also carry the".
