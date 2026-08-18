from openai import OpenAI

COLAB_LOCALTUNNEL_URL = "https://gold-files-pump.loca.lt/v1"

# Initialize the OpenAI client with the Localtunnel bypass header
client = OpenAI(
    base_url=COLAB_LOCALTUNNEL_URL,
    api_key="not-needed-for-local-vllm",
    default_headers={
        "Bypass-Tunnel-Reminder": "true"
    },  # <-- This bypasses the landing page block
)


def analyze_problem(user_problem_statement: str):
  response = client.chat.completions.create(
      model="meta-llama/Llama-3.1-8B-Instruct",
      messages=[
          {
              "role": "system",
              "content": (
                  "You are the Problem Analyzer agent for a Q-HPC system."
                  " Extract features, dataset characteristics, and constraints"
                  " from the user input into a structured JSON format."
              ),
          },
          {"role": "user", "content": user_problem_statement},
      ],
      temperature=0.0,
      response_format={"type": "json_object"},
  )
  return response.choices[0].message.content


if __name__ == "__main__":
  sample_problem = (
      "Classify a financial fraud dataset containing 50,000 tabular rows with"
      " 30 continuous features."
  )
  print(
      "Sending problem from Mac M4 Air to Colab GPU via Localtunnel (with"
      " bypass header)..."
  )
  structured_output = analyze_problem(sample_problem)
  print("\nStructured Output Received from Colab:")
  print(structured_output)