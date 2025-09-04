from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def summarize_with_chatgpt(results):
    summary_prompt = (
        "Oto wyniki klasyfikacji uszkodzeń samochodów:\n\n"
    )
    for r in results:
        summary_prompt += f"- {r['filename']}: {r['label']} ({r['probability']})\n"
    
    summary_prompt += (
        "\nPodsumuj te wyniki w sposób czytelny dla człowieka i dodaj krótką sugestię lub komentarz. Jeżeli to możliwe, uwzględnij informacje o uszkodzeniach pojazdów oraz ich markach. Wyniki zwróć w postaci JSON zawierający, label, probability, car brand oraz twoje krótkie podsumowanie\n"
    )

    print(summary_prompt)

    # Invoke
    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[
            {"role": "system", "content": "Jesteś ekspertem od analizy obrazu pojazdów pracującym dla firmy ubezpieczeniowej. Znasz się również na markach samochodów i ich uszkodzeniach."},
            {"role": "user", "content": [
                {'type': 'text', 'text': summary_prompt},
                {'type': "image_url", 'image_url': {'url': results[0]['original_image']}}
            ]}
        ],
        max_completion_tokens=300
    )

    return response.choices[0].message.content.strip()