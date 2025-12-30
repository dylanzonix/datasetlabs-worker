from openai import OpenAI
import json

client = OpenAI()

system_prompt = """You are a dataset row generator. Your job is to generate a single row for a dataset.

You have access to tools to help you gather information. Use them if needed. When you have everything you need, call generate_row with the final content.

## Row Instructions
<row_instructions>
{row_instructions}
</row_instructions>

## Column Schema
<column_schema>
{column_schema}
</column_schema>

## Seed
This is your starting point — source material to build from:
<seed>
{seed}
</seed>

## Diversity Targets
The row should have this flavor:
<diversity_targets>
{diversity_targets}
</diversity_targets>

## Your Task
Use the tools to gather what you need, then call generate_row when ready. You may call generate_row immediately if the seed is sufficient."""

tools = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search source documents for relevant information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_row",
            "description": "Generate the final row. Call this when you have everything you need.",
            "parameters": {
                "type": "object",
                "properties": {
                    "row": {"type": "object", "description": "The complete row matching the column schema"}
                },
                "required": ["row"]
            }
        }
    }
]


def mock_tool_call(name: str, args: dict) -> str:
    """Mock tool responses for testing."""
    if name == "rag_search":
        return json.dumps({
            "results": [
                {"text": "Mafia bosses typically maintain a cold, intimidating demeanor but follow strict codes of loyalty.", "chunk_id": "chunk_12"},
                {"text": "Common mafia greetings involve subtle shows of respect and power assessment.", "chunk_id": "chunk_45"}
            ]
        })
    elif name == "web_search":
        return json.dumps({
            "results": [
                {"title": "Mafia hierarchy explained", "snippet": "The boss sits at the top, followed by underboss, consigliere, and capos."}
            ]
        })
    elif name == "generate_row":
        return "ROW_COMPLETE"
    return "{}"


def run_agent(seed: str, diversity_targets: dict, row_instructions: str, column_schema: str):
    messages = [
        {
            "role": "system",
            "content": system_prompt.format(
                row_instructions=row_instructions,
                column_schema=column_schema,
                seed=seed,
                diversity_targets=json.dumps(diversity_targets, indent=2)
            )
        }
    ]
    
    max_iterations = 10
    
    for i in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                print(f"\n[TOOL CALL] {name}")
                print(f"  Args: {json.dumps(args, indent=2)[:200]}")
                
                if name == "generate_row":
                    print(f"\n[FINAL ROW]")
                    print(json.dumps(args.get("row", args), indent=2))
                    return args.get("row", args), messages
                
                result = mock_tool_call(name, args)
                print(f"  Result: {result[:200]}...")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            print(f"\n[NO TOOL CALL] Agent response: {message.content[:200] if message.content else 'None'}")
            break
    
    return None, messages


if __name__ == "__main__":
    seed = """*His name is Damian, he's a dangerous criminal knows also as a mafia boss in the town.
You just moved in the town and wanted to know your neighbors, it turned out your neighbor was Damian. 
You walked to his door with own made cookies and knocked on his door, when he opened you saw a not shaved, 
handsome man with eyebags that looked like he didn't sleep for weeks.*
"who are you and what do you want. I'm kinda busy here." *he says with a deep and dark morning voice looking down at you.*"""

    diversity_targets = {
        "genre": "crime",
        "tone": "dark"
    }
    
    row_instructions = """Each row is a multi-turn roleplay conversation between a user and an AI character. 
The conversation should feel natural and immersive. The AI character has a defined persona established in the system message. 
Conversations involve back-and-forth exchanges where the user and AI engage in a scenario together."""
    
    column_schema = """conversation (array of message objects):
  - from: "system" | "human" | "gpt"
  - value: string (the message content)

Structure:
- First message is "system" defining the AI character/persona
- Alternating "human" and "gpt" messages follow
- Minimum 2 turns (4 messages after system)
- Conversations should have natural flow and stay in character"""

    row, trace = run_agent(seed, diversity_targets, row_instructions, column_schema)