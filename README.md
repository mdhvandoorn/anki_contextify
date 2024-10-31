# Anki Contextify: Add Illustrative Examples and Conjugations to Words in an Anki Deck
This script takes a "Notes in Plain Text" export from an Anki deck and enriches a specified field containing a word by adding:
1. Illustrative example sentences that showcase the different meanings of the word.
2. Conjugations for specified tenses, in the case of verbs.

The OpenAI API is used to generate these enrichments.

## Process
1. Read the txt file that contains the original deck.
2. Extract the words from a specified field.
3. Flag words that are also verbs.
4. Clean the words.
5. Read the prompt that is used to query OpenAI.
6. Get the Chain-of-Thought messages that help specify the desired outcome of the prompt.
7. Get response per word.
8. Separate the examples and conjugations components from the responses.
9. Retrieve the final conjugations. Keep only regular conjugations and format them in HTML.
10. Retrieve the final examples and format them in HTML.
11. Replace the original word-field content with the enriched version, which now includes the word along with examples and, if applicable, conjugations.
12. Export the result as a txt file, which can be imported directly into Anki to update the specified field without altering the other fields.

## Performance
Beware that querying OpenAI can take a while. The [Essential Spanish Vocabulary Top 5000](https://ankiweb.net/shared/info/241428882) deck contains 5000 notes, which results in 5000 API requests. As of 31-10-2024 this took 111 minutes with the 4o-mini model. 

# Feedback, Questions, and Contributions
This is a quick side project born out of a need to automate the enrichment of the [Essential Spanish Vocabulary Top 5000](https://ankiweb.net/shared/info/241428882) deck. As such, it’s currently not generalized for other decks, and some parts of the process could be more efficient. Feel free to suggest improvements or reach out with feedback/questions!
