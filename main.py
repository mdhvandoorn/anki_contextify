import difflib
import pandas as pd
import os
import re
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI
from pprint import pprint

import csv

PROMPT_DIR = os.path.dirname(os.path.abspath(__file__))

LOC_PROMPT = os.path.join(PROMPT_DIR, "prompt_v4.txt")

LOC_EX_1_IN = os.path.join(PROMPT_DIR, "prompt_v4_ex1_in.txt")
LOC_EX_1_OUT = os.path.join(PROMPT_DIR, "prompt_v4_ex1_out.txt")

LOC_EX_2_IN = os.path.join(PROMPT_DIR, "prompt_v4_ex2_in.txt")
LOC_EX_2_OUT = os.path.join(PROMPT_DIR, "prompt_v4_ex2_out.txt")

# or None
N_ROWS = None

PRESENT_INDICATIVE_CONJUG = {
    "ar": {
        "yo": "o",
        "tú": "as",
        "él": "a",
        "nosotros": "amos",
        "vosotros": "áis",
        "ellas": "an",
    },
    "er": {
        "yo": "o",
        "tú": "es",
        "él": "e",
        "nosotros": "emos",
        "vosotros": "éis",
        "ellas": "en",
    },
    "ir": {
        "yo": "o",
        "tú": "es",
        "él": "e",
        "nosotros": "imos",
        "vosotros": "ís",
        "ellas": "en",
    },
}

PRETERITE_INDICATIVE_CONJUG = {
    "ar": {
        "yo": "é",
        "tú": "aste",
        "él": "ó",
        "nosotros": "amos",
        "vosotros": "asteis",
        "ellas": "aron",
    },
    "er": {
        "yo": "í",
        "tú": "iste",
        "él": "ió",
        "nosotros": "imos",
        "vosotros": "isteis",
        "ellas": "ieron",
    },
    "ir": {
        "yo": "í",
        "tú": "iste",
        "él": "ió",
        "nosotros": "imos",
        "vosotros": "isteis",
        "ellas": "ieron",
    },
}

IMPERFECT_INDICATIVE_CONJUG = {
    "ar": {
        "yo": "aba",
        "tú": "abas",
        "él": "aba",
        "nosotros": "ábamos",
        "vosotros": "abais",
        "ellas": "aban",
    },
    "er": {
        "yo": "ía",
        "tú": "ías",
        "él": "ía",
        "nosotros": "íamos",
        "vosotros": "íais",
        "ellas": "ían",
    },
    "ir": {
        "yo": "ía",
        "tú": "ías",
        "él": "ía",
        "nosotros": "íamos",
        "vosotros": "íais",
        "ellas": "ían",
    },
}

CONJUGATIONS = {
    "PRESENT_INDICATIVE_CONJUG": PRESENT_INDICATIVE_CONJUG,
    "PRETERITE_INDICATIVE_CONJUG": PRETERITE_INDICATIVE_CONJUG,
    "IMPERFECT_INDICATIVE_CONJUG": IMPERFECT_INDICATIVE_CONJUG,
}

# take environment variables from .env.
load_dotenv()

SOURCE_FIELD = 3
TARGET_FIELD = 3


def main():
    start_time = time.time()

    arg_names = ["input_file", "output_file"]

    # Check if the correct number of arguments is provided
    if len(sys.argv) != len(arg_names) + 1:
        usage_str = "Usage: python main.py " + " ".join([f"<{arg}>" for arg in arg_names])
        print(usage_str)
        sys.exit(1)

    print("Arguments received:")
    for i, arg in enumerate(sys.argv[1:], 1):
        print(f"Argument {i} ({arg_names[i-1]}): {arg}")

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    api_key = os.getenv("OPENAI_API_KEY")

    os.environ["OPENAI_API_KEY"] = api_key

    client = OpenAI()

    org_notes = pd.read_csv(
        input_file, sep="\t", skiprows=6, header=None, nrows=N_ROWS
    )

    notes = org_notes.copy(deep=True)

    # The Spanish field contains "Key to Abbreviations" for a card. It is not
    # like the cards that contain words that are supposed to be learned.
    # It is an exception with unknown rationale and should be ignored.
    notes = notes[notes[SOURCE_FIELD] != "Key to Abbreviations"]
    # Use regular expression to capture everything up to the first ")"
    notes["org_word"] = notes[SOURCE_FIELD].str.extract(r"([^)]*\))")

    # Add a verb flag column (True if org_word contains '(v)')
    notes["verb_flag"] = notes["org_word"].apply(lambda x: "(v)" in x)

    # Add a clean_word column which excludes the parentheses part
    notes["clean_word"] = notes["org_word"].apply(lambda x: x.split(" (")[0])

    prompt = get_prompt(LOC_PROMPT)

    cot_messages = get_cot_messages(prompt)

    notes = get_responses(notes, client, prompt, cot_messages)

    notes[["examples", "conjugations"]] = notes.apply(
        lambda row: extract_context_components(row["with_context"]),
        axis=1,
    ).apply(pd.Series)

    final_conjug = get_final_verb_conjugations(notes)

    notes["final_conjugations"] = final_conjug

    notes["final_examples"] = notes["examples"].apply(get_html_examples)

    notes["final_html"] = (
        notes["org_word"]
        + notes["final_examples"]
        + notes["final_conjugations"].fillna("")
    )

    org_notes[SOURCE_FIELD] = notes["final_html"]

    # quoting=csv.QUOTE_NONE to prevent insertion of quotation marks as this
    # prohibits expected html interpretation
    org_notes.to_csv(
        "contexified.txt",
        sep="\t",
        index=False,
        header=False,
        quoting=csv.QUOTE_NONE,
    )

    # Read the first six lines from original export file
    with open(input_file, "r") as file1:
        first_six_lines = [next(file1) for _ in range(6)]

    # Get the content of the contextified file
    with open("contexified.txt", "r") as file2:
        file2_content = file2.readlines()

    # Append and write final contextified deck
    combined_content = first_six_lines + file2_content
    with open(output_file, "w") as output_file:
        output_file.writelines(combined_content)

    total_in_tokens = notes["in_tokens"].sum()
    total_out_tokens = notes["out_tokens"].sum()

    end_time = time.time()

    exec_time_minutes = (end_time - start_time) / 60

    print(
        f"Contextified deck saved to: final.txt\nTotal input tokens: {total_in_tokens}; total output tokens: {total_out_tokens}.This took {exec_time_minutes} minutes.\n"
    )


def get_responses(
    df: pd.DataFrame, client: OpenAI, prompt: str, cot_messages: list
) -> pd.DataFrame:
    org_prompt = prompt
    responses = []
    for count, (index, row) in enumerate(df.iterrows(), start=1):
        if count % 100 == 0:
            print(f"Processing note {count} ({round(count/df.shape[0],1)}%)")
        org_word = row["org_word"]
        prompt = org_prompt
        prompt = insert_org_word(prompt, org_word)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.5,
            messages=cot_messages
            + [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        )
        response = completion.choices[0].message.content
        in_tokens = completion.usage.prompt_tokens
        out_tokens = completion.usage.completion_tokens
        # Call the existing get_response function
        # response, in_tokens, out_tokens = get_response(org_word, client, prompt, cot_messages)
        responses.append((response, in_tokens, out_tokens))
    # Convert the list of tuples into a DataFrame
    response_df = pd.DataFrame(
        responses, columns=["with_context", "in_tokens", "out_tokens"]
    )
    # Concatenate the original DataFrame with the new response DataFrame
    df = pd.concat([df.reset_index(drop=True), response_df], axis=1)
    return df


def get_html_examples(examples: str) -> str:

    # Add 34 <br> tags at the beginning
    br_tags = "<br>" * 34

    # Start building the HTML output
    html_output = (
        br_tags + '<div id="exampleSentences"><b>Example Sentences</b><br>'
    )

    # Process the text:
    # - Strip leading/trailing whitespace
    # - Split into lines
    # - Remove empty lines
    # - Join lines with <br>
    lines = examples.strip().split("\n")
    lines = [
        line.strip() for line in lines if line.strip()
    ]  # Remove empty lines and strip each line
    formatted_text = "<br><br>".join(lines)

    # Add the formatted text to the HTML output
    html_output += formatted_text

    # Close the div and add the final <br><b></b>
    html_output += "</div><br>"

    return html_output


def create_final_conjug_html(conj_dict):
    # Define the order of subjects and tenses
    subjects = ["yo", "tú", "él", "nosotros", "vosotros", "ellas"]
    tenses = [
        "Indicative Present",
        "Indicative Preterite",
        "Indicative Imperfect",
    ]
    tense_keys = {
        "Indicative Present": "indicative_present",
        "Indicative Preterite": "indicative_preterite",
        "Indicative Imperfect": "indicative_imperfect",
    }
    tense_ids = {
        "Indicative Present": "indicativePresent",
        "Indicative Preterite": "indicativePreterite",
        "Indicative Imperfect": "indicativeImperfect",
    }

    # Start building the HTML content
    html_lines = []
    html_lines.append(
        '<div id="verbConjugations"><b>Verb Conjugations</b><br>'
    )
    first_tense = True  # Flag to check if it's the first tense

    for tense_name in tenses:
        tense_key = tense_keys[tense_name]
        tense_id = tense_ids[tense_name]

        if not first_tense:
            html_lines.append("<br>")
        first_tense = False

        html_lines.append(f'<div id="{tense_id}"><u>{tense_name}</u><br>')

        if tense_key in conj_dict and conj_dict[tense_key]:
            tense_data = conj_dict[tense_key]
            missing_sequence = False
            entries_added = False  # Flag to check if any entries were added
            for subj in subjects:
                if subj in tense_data:
                    entries_added = True
                    if missing_sequence:
                        # If there was a sequence of missing subjects, add a '-'
                        html_lines.append("-<br>")
                        missing_sequence = False
                    # Get the conjugation and indices
                    conjugation, indices = tense_data[subj]
                    # Mark letters at specified indices in red
                    conjugation_marked = ""
                    for i, c in enumerate(conjugation):
                        if i in indices:
                            conjugation_marked += (
                                f'<span style="color: red;">{c}</span>'
                            )
                        else:
                            conjugation_marked += c
                    # Add the subject and conjugation
                    html_lines.append(f"{subj} {conjugation_marked}<br>")
                else:
                    missing_sequence = True
            # Handle missing subjects at the end
            if missing_sequence and html_lines[-1] != "-<br>":
                html_lines.append("-<br>")
            html_lines.append("</div>")
        else:
            # If the entire tense is missing or conj_dict is empty
            html_lines.append("-</div>")

    html_lines.append("</div>")
    # Join the lines into a single string
    final_conjug_html = "".join(html_lines)
    return final_conjug_html


def get_final_verb_conjugations(df: pd.DataFrame) -> pd.DataFrame:
    final_conjug = []

    for index, row in df.iterrows():
        if row["verb_flag"]:
            expected_conjug = get_expected_conjugation(row["clean_word"])
            actual_conjug = llm_conjug_to_dict(row["conjugations"])
            deviations = find_deviations(expected_conjug, actual_conjug)
            final_conjug_txt = create_final_conjug_html(deviations)

            final_conjug.append(final_conjug_txt)
        else:
            final_conjug.append(None)

    return final_conjug


def get_deviation_indices(expected, actual):
    s = difflib.SequenceMatcher(None, expected, actual)
    indices = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "replace" or tag == "delete":
            indices.extend(range(i1, i2))
        elif tag == "insert":
            # For insertions, we consider the position where the insertion occurs
            indices.append(i1)
    return sorted(set(indices))


def find_deviations(expected_conjugations, actual_conjugations):
    deviations = {}

    for tense, subjects in actual_conjugations.items():
        if tense in expected_conjugations:
            tense_deviations = {}
            for subject, actual_form in subjects.items():
                expected_form = expected_conjugations[tense].get(subject)
                if actual_form != expected_form:  # Compare actual vs expected
                    deviating_indices = get_deviation_indices(
                        expected_form, actual_form
                    )
                    tense_deviations[subject] = [
                        actual_form,
                        deviating_indices,
                    ]

            if tense_deviations:  # Only keep tenses with deviations
                deviations[tense] = tense_deviations

    return deviations


def llm_conjug_to_dict(text):
    # Split the text into sections by tense
    sections = text.split("\n\n")

    conjugation_dict = {}

    for section in sections:
        lines = section.split("\n")
        tense_name = (
            lines[0].strip().lower().replace(" ", "_")
        )  # Convert tense name to match desired format
        conjugation_dict[tense_name] = {}

        for line in lines[1:]:
            parts = line.split()
            subject = parts[0]
            verb_form = parts[1]
            conjugation_dict[tense_name][subject] = verb_form

    return conjugation_dict


def get_expected_conjugation(verb: str) -> dict:
    root = verb[:-2]

    if verb.endswith("ar"):
        verb_type = "ar"
    elif verb.endswith("er"):
        verb_type = "er"
    elif verb.endswith("ir"):
        verb_type = "ir"
    else:
        return None  # Not a valid infinitive verb

    # Generate expected_conjugation
    expected_conjugation = {
        "indicative_present": {
            subject: root
            + CONJUGATIONS["PRESENT_INDICATIVE_CONJUG"][verb_type][subject]
            for subject in CONJUGATIONS["PRESENT_INDICATIVE_CONJUG"][verb_type]
        },
        "indicative_preterite": {
            subject: root
            + CONJUGATIONS["PRETERITE_INDICATIVE_CONJUG"][verb_type][subject]
            for subject in CONJUGATIONS["PRETERITE_INDICATIVE_CONJUG"][
                verb_type
            ]
        },
        "indicative_imperfect": {
            subject: root
            + CONJUGATIONS["IMPERFECT_INDICATIVE_CONJUG"][verb_type][subject]
            for subject in CONJUGATIONS["IMPERFECT_INDICATIVE_CONJUG"][
                verb_type
            ]
        },
    }

    return expected_conjugation


def extract_context_components(context: str) -> tuple[str, set[None, str]]:
    # Define the regular expressions for the two components
    examples_pattern = r"EXAMPLES ####(.*?)####"
    conjugations_pattern = r"CONJUGATIONS \$\$\$\$(.*?)\$\$\$\$"

    # Extract the components using re.search and the defined patterns
    examples_match = re.search(examples_pattern, context, re.DOTALL)
    conjugations_match = re.search(conjugations_pattern, context, re.DOTALL)

    # Extract the matched content, or return None if not found
    examples = examples_match.group(1).strip() if examples_match else None
    conjugations = (
        conjugations_match.group(1).strip() if conjugations_match else None
    )

    return examples, conjugations


def get_prompt(loc_prompt: str) -> str:

    # Open and read prompt
    with open(loc_prompt, "r") as file:
        prompt = file.read()

    return prompt


def get_cot_messages(prompt: str) -> list:

    # Open and read prompt
    with open(LOC_EX_1_IN, "r") as file:
        ex1_in = file.read()

    with open(LOC_EX_1_OUT, "r") as file:
        ex1_out = file.read()

    # Open and read prompt
    with open(LOC_EX_2_IN, "r") as file:
        ex2_in = file.read()

    with open(LOC_EX_2_OUT, "r") as file:
        ex2_out = file.read()

    ex_messages = [
        {
            "role": "system",
            "content": "You provide context for Spanish words. Your responses exactly obey the required formatting.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": insert_org_word(prompt, ex1_in)}
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": ex1_out}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": insert_org_word(prompt, ex2_in)}
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": ex2_out}]},
    ]

    return ex_messages


def insert_org_word(prompt, org_word):
    # Find the index of the last occurrence of the target word
    index = prompt.rfind("word:")

    # If the word is found, insert the new word after it
    if index != -1:
        # Calculate the position after the target word
        insertion_point = index + len("word:")
        # Insert the org_word after the target word
        modified_text = (
            prompt[:insertion_point]
            + " "
            + org_word
            + prompt[insertion_point:]
        )
        return modified_text
    else:
        # If the word is not found, raise error
        raise RuntimeError("Could not insert org_word in prompt")


if __name__ == "__main__":
    main()
