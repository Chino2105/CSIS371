#import os

#from dotenv import load_dotenv
#from openai import OpenAI

# File: gpt.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 11/5/2024
# Description: This program serves as a wrapper for OpenAI's GPT API to facilitate

#load_dotenv() # Load Envirmonemtal Vars

#client = OpenAI(api_key=os.getenv("GPT_KEY")) # Load OpenAI

#api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_KEY")

#if not api_key:
    # No key set – don't create a client yet
 #   client = None
#else:
 #   client = OpenAI(api_key=api_key)




#def sendGPT(text):
   
 #   if client is None:
  #          raise RuntimeError(
   #         "OpenAI client not initialized. "
 #           "Set OPENAI_API_KEY or GPT_KEY in your environment or .env file."
  #      )

   # try:
    ##    response = client.responses.create(
      #      model="gpt-5-nano",  # you can change this to a bigger model if you want
    # input=text,
     #       max_output_tokens=400,
   #     )
        # Using the helper property your version of the SDK provides
      #  return response.output_text

   # except Exception as e:
    #    raise RuntimeError(f"OpenAI API error: {e}") from e
    
    
   # return response.output_text
# gpt.py
# Simple wrapper around OpenAI Responses API for the ToT shell.

# gpt.py
# Simple wrapper around OpenAI Chat Completions API for the ToT shell.

from openai import OpenAI

# >>> PUT YOUR REAL KEY HERE <<<
client = OpenAI(api_key="sk-NEEDKEY")


def sendGPT(text: str) -> str:
    """
    Send text to GPT and return the plain-text response.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",  # or "gpt-5.1-mini" if your key has access
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise helper inside an information retrieval shell. "
                        "Explain clearly and briefly; bullet points are fine."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=400,
        )

        reply = resp.choices[0].message.content or ""
        return reply.strip()

    except Exception as e:
        # Let the caller print a clean error
        raise RuntimeError(f"OpenAI API error: {e}") from e


if __name__ == "__main__":
    print(sendGPT("Say 'hello from the ToT helper' in one short sentence."))
