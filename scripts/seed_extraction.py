from openai import OpenAI
from pydantic import BaseModel, ValidationError
import json
import re

client = OpenAI()

# Pydantic schemas
class SeedMarker(BaseModel):
    start: str
    end: str

class ExtractionResponse(BaseModel):
    seeds: list[SeedMarker]

# Seed extraction from chunk
def extract_seed_text(chunk: str, marker: SeedMarker) -> str | None:
    """
    Find the continuous span between start and end markers in the chunk.
    Returns the full text including start and end, or None if not found.
    """
    start_idx = chunk.find(marker.start)
    if start_idx == -1:
        return None
    
    end_idx = chunk.find(marker.end, start_idx)
    if end_idx == -1:
        return None
    
    # Include the end marker in the result
    return chunk[start_idx:end_idx + len(marker.end)]

def process_extraction_response(raw_response: str, chunk: str) -> list[str]:
    """
    Validate LLM response and extract seed texts from chunk.
    Returns list of extracted seed strings.
    """
    # Parse JSON
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return []
    
    # Validate with Pydantic
    try:
        response = ExtractionResponse(**data)
    except ValidationError as e:
        print(f"Validation error: {e}")
        return []
    
    # Extract seed texts
    extracted_seeds = []
    for marker in response.seeds:
        seed_text = extract_seed_text(chunk, marker)
        if seed_text:
            extracted_seeds.append(seed_text)
        else:
            print(f"Warning: Could not locate seed with start='{marker.start[:30]}...'")
    
    return extracted_seeds

# Main execution
row_instructions = """Each row is a multi-turn roleplay conversation between a user and an AI character. The conversation should feel natural and immersive. The AI character has a defined persona established in the system message. Conversations involve back-and-forth exchanges where the user and AI engage in a scenario together."""

column_schema = """conversation (array of message objects):
  - from: "system" | "human" | "gpt"
  - value: string (the message content)

Structure:
- First message is "system" defining the AI character/persona
- Alternating "human" and "gpt" messages follow
- Minimum 2 turns (4 messages after system)
- Conversations should have natural flow and stay in character"""

chunks = [
    '''bot_uid,name,image_url,developer_uid,conversations,description,engagement,median_nonzero_chat,messages_sent,score,first_message
_bot_f3e6eb63-69c2-476d-8c3a-07711b07b04e,Jayden (enemy agent),http://images.chai.ml/bots%2FmV6UWsn8uCaT0Na9TWsvLmNZMLp2%2F1728143844839.jpg?alt=media&token=b92883ec-e3f0-4f56-bdb8-530597827d68,mV6UWsn8uCaT0Na9TWsvLmNZMLp2,215197,ur silly enemy agent/target for the gala mssn,,27,5633568,,"You and Jayden are both agents for different organizations. He works for The New Day, his code name is ND 471. You work for The Last Night, your code name is LN 201. Both of you are the best on your team. Both of your teams are looking to 'eliminate' the competition, starting with their best agent. Your goal is to eliminate him.

*You get to the gala—the spot of the mission—rather late. Most people are already there, Including Jayden. He spots you across the room and smirks. Who will die first?*
"''',
    '''bot_uid,name,image_url,developer_uid,conversations,description,engagement,median_nonzero_chat,messages_sent,score,first_message
_bot_89cf0ab5-a58a-4a8d-b6e1-9abe72f6463a,Damian (Mafia Boss Neighbor),http://images.chai.ml/bots%2Fz4bXEiGzCdcAjSTOcgOqODUMC7D3%2F1721544492186.jpg?alt=media&token=07d99e89-c9ad-4f23-bb44-f0b4a48d8f37,z4bXEiGzCdcAjSTOcgOqODUMC7D3,379690,Dangerous Mafia Boss,,28,13570549,,"
*His name is Damian, he's a dangerous criminal knows also as a mafia boss in the town.
You just moved in the town and wanted to know your neighbors,it turned out your neighbor was Damian. You walked to his door with own made cookies and knocked on his door,when he opened you saw a not shaved, handsome man with eyebags that looked like he didn't sleep for weeks.*
""who are you and what do you want. I'm kinda busy here."" *he says with a deep and dark morning voice looking down at you.*"''',
    '''bot_uid,name,image_url,developer_uid,conversations,description,engagement,median_nonzero_chat,messages_sent,score,first_message
_bot_968ecee9-3dc9-4357-bbf6-b7defe90902a,"Ральф (твой сын, глава мафии, ты объявилась спустя 16 лет)",http://images.chai.ml/bots%2FZPY1IONW76bVB2sUYGa5hd2B7a62%2F1740252751906.jpg?alt=media&token=dd5c2b9d-137d-4944-a410-9040960d1c55,ZPY1IONW76bVB2sUYGa5hd2B7a62,881,,,28,19558,,"*У тебя была сложная жизнь.... Мало того что у тебя была строгая мафиозная семья ты ещё и как то забеременела в 16 лет, и уговорила свою мать присмотреть за будущим ребёнком пока ты уедешь устраивать личную жизнь... Через 16 лет у тебя уже был прибыльный бизнес и ты нашла окошко что бы приехать навестить родителей и своего уже взрослого сына который даже не знал тебя... Ты постучала в дверь и тебе открыла твоя мать, кажется она не слишком обрадовалась но впустила тебя в дом*"'''
]

for i, chunk in enumerate(chunks):
    print(f"\n{'='*50}")
    print(f"Chunk {i+1}")
    print('='*50)
    
    response = client.responses.create(
        prompt={
            "id": "pmpt_69508e29f514819693d017e0848e223406fd27a87843182b",
            "version": "5",
            "variables": {
                "row_instructions": row_instructions,
                "column_schema": column_schema,
                "source_chunk": chunk
            }
        },
        input=[],
        reasoning={"summary": "auto"},
        store=True,
    )
    
    raw_output = response.output_text
    print(f"\nLLM Response:\n{raw_output}")
    
    seeds = process_extraction_response(raw_output, chunk)
    
    print(f"\nExtracted Seeds ({len(seeds)}):")
    for j, seed in enumerate(seeds):
        print(f"\n--- Seed {j+1} ---")
        print(seed[:2000] + "..." if len(seed) > 2000 else seed)