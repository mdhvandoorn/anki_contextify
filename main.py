import pandas as pd
import os

from dotenv import load_dotenv
from openai import OpenAI

# take environment variables from .env.
load_dotenv()  


SOURCE_FIELD = 3
TARGET_FIELD = 3

def main():
    api_key = os.getenv('OPENAI_API_KEY')

    os.environ['OPENAI_API_KEY'] = api_key
    
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Write a haiku about recursion in programming."
            }
        ]
    )

    print(completion.choices[0].message)
        

    notes = pd.read_csv('test_export.txt', sep='\t', skiprows=6, header=None)

    print(notes[SOURCE_FIELD])

    


if __name__ == "__main__":
    main()