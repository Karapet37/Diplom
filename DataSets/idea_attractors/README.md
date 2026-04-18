# Idea Attractors Dataset

This folder contains a seed dataset of short sayings, parables, fables, slogans, thought-stoppers, and false or harmful rallying ideas that people use to coordinate, moralize, simplify complexity, or form group identity.

It is intentionally mixed. Some records capture durable folk wisdom, some capture movement language, and some capture manipulative or conspiratorial phrases included for analysis rather than endorsement.

## Files

- `idea_attractors_seed.jsonl`: curated JSONL corpus with one record per line.

## Schema

Each JSONL record has these fields:

- `id`: stable record identifier.
- `text`: exact short phrase, title, or belief claim.
- `form`: one of `proverb`, `parable`, `fable`, `slogan`, `catchphrase`, or `belief_claim`.
- `cluster`: broad family such as `folk_wisdom`, `religious_story`, `classical_fable`, `movement_politics`, `thought_stopper`, or `conspiracy`.
- `source_mode`: either `exact_phrase` or `title_plus_curated_summary`.
- `quality_label`: rough interpretive label.
- `crowd_pull`: list of mechanisms that help the item spread or gather people.
- `summary`: short explanation of the idea or the role it plays.
- `source_name`: source page or source collection used for curation.
- `source_url`: source link for provenance.

## Quality Labels

- `wise`: durable practical or moral guidance.
- `mixed`: partially useful but highly context-dependent.
- `mobilizing`: mainly a rallying frame for collective action or identity.
- `misleading`: catchy language that tends to oversimplify or shut down thought.
- `false_or_harmful`: factually false, dehumanizing, conspiratorial, or dangerous.

## Notes

- Exact wording is preserved only for short phrases, titles, slogans, and other brief expressions.
- Longer stories are represented by title plus a curated summary rather than full text.
- Harmful items are included to support analysis of group formation, propaganda, and social contagion.
- This is a seed corpus, not an exhaustive catalog of all such expressions.

## Source Set

- Wiktionary Appendix: English proverbs
  - `https://en.wiktionary.org/wiki/Appendix:English_proverbs`
- Wikipedia: Parables of Jesus
  - `https://en.wikipedia.org/wiki/Parables_of_Jesus`
- Wikipedia: List of Aesop's Fables
  - `https://en.wikipedia.org/wiki/List_of_Aesop%27s_Fables`
- Wikipedia: Panchatantra
  - `https://en.wikipedia.org/wiki/Panchatantra`
- Wikipedia: List of political slogans
  - `https://en.wikipedia.org/wiki/List_of_political_slogans`
- Wikipedia: Power to the people (slogan)
  - `https://en.wikipedia.org/wiki/Power_to_the_people_(slogan)`
- Wikipedia: One man, one vote
  - `https://en.wikipedia.org/wiki/One_man,_one_vote`
- Wikipedia: Thought-terminating cliche
  - `https://en.wikipedia.org/wiki/Thought-terminating_clich%C3%A9`
- Wikipedia: QAnon
  - `https://en.wikipedia.org/wiki/QAnon`
- PMC / Synthese: Do your own research!
  - `https://pmc.ncbi.nlm.nih.gov/articles/PMC9392429/`
- Wikipedia: Flat Earth
  - `https://en.wikipedia.org/wiki/Flat_Earth`
- Wikipedia: Great Replacement conspiracy theory
  - `https://en.wikipedia.org/wiki/Great_Replacement_conspiracy_theory`
