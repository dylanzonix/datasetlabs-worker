import numpy as np
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from openai import OpenAI

client = OpenAI()


@dataclass
class SelectedSeed:
    id: str
    text: str
    assigned: Dict[str, str]  # {axis_name: value}
    source_chunk_id: str


@dataclass
class Chunk:
    id: str
    text: str
    embedding: np.ndarray


@dataclass
class Ingredient:
    question: str
    status: str  # "resolved" | "partial" | "unresolved"
    answer: Optional[str]
    source_chunk_id: Optional[str]


@dataclass
class Recipe:
    seed_id: str
    seed_text: str
    assigned: Dict[str, str]
    completeness: float
    ingredients: List[Ingredient]


# =============================================================================
# STEP 1: Analyze seed to get completeness + sub-questions
# =============================================================================

ANALYSIS_PROMPT = """You are analyzing a seed for synthetic dataset generation.

## Dataset Context

<row_instructions>
{row_instructions}
</row_instructions>

<column_schema>
{column_schema}
</column_schema>

## Assigned Diversity Values

{assigned_values}

## Seed Text

<seed>
{seed_text}
</seed>

## Your Task

Analyze this seed and determine:

1. **Completeness (0.0 - 1.0):** How much of the information needed to generate a complete row is already present in the seed? 
   - 1.0 = seed contains everything needed
   - 0.5 = seed contains about half of what's needed
   - 0.1 = seed is just a starting point, most info is missing

2. **Sub-questions:** What specific questions need to be answered to fill in the gaps? These will be used to search for additional context.
   - Only include questions where the answer is NOT in the seed
   - Be specific — these become search queries
   - If completeness is 1.0, return empty list

## Output Format

Return JSON only. No explanation.

{
  "completeness": 0.0-1.0,
  "sub_questions": ["question 1", "question 2", ...]
}
"""


def analyze_seed(
    seed: SelectedSeed,
    row_instructions: str,
    column_schema: str
) -> Dict:
    """
    Analyze a seed to determine completeness and sub-questions.
    
    ============================================================================
    PRODUCTION NOTES
    ============================================================================
    
    BATCHING:
    If processing many seeds, batch the analysis calls:
    
        async def analyze_seeds_batch(seeds: List[SelectedSeed], ...):
            tasks = [analyze_seed_async(seed, ...) for seed in seeds]
            return await asyncio.gather(*tasks)
    
    Or use OpenAI's batch API for cost savings on large jobs.
    
    CACHING:
    If same seed is re-analyzed (e.g., job resume), cache results:
    
        cache_key = hash(seed.id + row_instructions + column_schema)
        if cache_key in redis:
            return redis.get(cache_key)
    
    ============================================================================
    """
    assigned_str = "\n".join(f"- {k}: {v}" for k, v in seed.assigned.items())
    
    prompt = ANALYSIS_PROMPT.format(
        row_instructions=row_instructions,
        column_schema=column_schema,
        assigned_values=assigned_str,
        seed_text=seed.text
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    result = json.loads(response.choices[0].message.content)
    return result


# =============================================================================
# STEP 2: RAG - Find relevant chunks for each sub-question
# =============================================================================

def compute_embedding(text: str) -> np.ndarray:
    """Embed a single text."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=[text],
        encoding_format="float"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rag_search(
    query: str,
    chunks: List[Chunk],
    top_k: int = 3,
    min_score: float = 0.0
) -> List[Chunk]:
    """
    Find top-k most relevant chunks for a query.
    
    ============================================================================
    PRODUCTION IMPLEMENTATION
    ============================================================================
    
    This in-memory version is for testing only. In production, query pgvector:
    
        from sqlalchemy import text
        
        def rag_search_pgvector(
            query: str,
            project_id: UUID,
            db_session: Session,
            top_k: int = 3,
            min_score: float = 0.3
        ) -> List[Chunk]:
            # Embed the query
            query_embedding = compute_embedding(query)
            
            # Query pgvector with cosine distance
            # Note: <=> is cosine distance, lower = more similar
            # Convert to similarity: 1 - distance
            sql = text('''
                SELECT 
                    id,
                    text,
                    1 - (embedding <=> :query_embedding) as similarity
                FROM rag_chunks
                WHERE project_id = :project_id
                  AND 1 - (embedding <=> :query_embedding) > :min_score
                ORDER BY embedding <=> :query_embedding
                LIMIT :top_k
            ''')
            
            results = db_session.execute(sql, {
                "query_embedding": query_embedding.tolist(),
                "project_id": project_id,
                "min_score": min_score,
                "top_k": top_k
            }).fetchall()
            
            return [
                Chunk(id=row.id, text=row.text, embedding=None)
                for row in results
            ]
    
    CONFIGURATION:
    - top_k: 3-5 is usually sufficient. More = more context but slower + costlier.
    - min_score: Filter out low-relevance chunks. 0.3 is a reasonable threshold.
      Set to 0.0 to disable filtering.
    
    OPTIMIZATION:
    If multiple sub-questions per seed, batch the embedding calls:
    
        def rag_search_batch(queries: List[str], ...) -> Dict[str, List[Chunk]]:
            # Embed all queries in one API call
            embeddings = compute_embeddings(queries)
            
            # Could also batch the pgvector queries using ANY()
            results = {}
            for query, emb in zip(queries, embeddings):
                results[query] = search_with_embedding(emb, ...)
            return results
    
    ============================================================================
    """
    query_embedding = compute_embedding(query)
    
    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk.embedding)
        if score >= min_score:
            scored.append((score, chunk))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]


# =============================================================================
# STEP 3: Resolve - Extract answers from RAG results
# =============================================================================

RESOLVE_PROMPT = """You are extracting an answer from source documents.

## Question

{question}

## Source Documents

{sources}

## Your Task

Based on the source documents, answer the question.

- If the sources clearly answer the question: status = "resolved"
- If the sources partially answer it: status = "partial"  
- If the sources don't help: status = "unresolved"

For resolved/partial, extract the relevant information as the answer.
Preserve original phrasing where possible — do not paraphrase unnecessarily.

## Output Format

Return JSON only. No explanation.

{{
  "status": "resolved" | "partial" | "unresolved",
  "answer": "extracted answer or null if unresolved",
  "source_chunk_id": "id of most relevant chunk or null"
}}
"""


def resolve_question(
    question: str,
    chunks: List[Chunk]
) -> Ingredient:
    """
    Given a question and relevant chunks, extract an answer.
    
    ============================================================================
    PRODUCTION NOTES
    ============================================================================
    
    HANDLING CONFLICTS:
    If multiple chunks provide conflicting information, the prompt could be
    extended to handle this:
    
        - If sources conflict, prefer the most recent or authoritative
        - Or: status = "conflicting", answer = summary of different claims
    
    BATCHING:
    Like analyze_seed, this can be batched across multiple questions:
    
        async def resolve_questions_batch(
            questions: List[str],
            chunks_per_question: List[List[Chunk]]
        ) -> List[Ingredient]:
            tasks = [
                resolve_question_async(q, c) 
                for q, c in zip(questions, chunks_per_question)
            ]
            return await asyncio.gather(*tasks)
    
    COST OPTIMIZATION:
    For large datasets, consider using gpt-4o-mini for resolution.
    Quality is slightly lower but 10x cheaper.
    Reserve gpt-4o for analysis step where reasoning matters more.
    
    ============================================================================
    """
    if not chunks:
        return Ingredient(
            question=question,
            status="unresolved",
            answer=None,
            source_chunk_id=None
        )
    
    sources_str = "\n\n".join(
        f"[Chunk {chunk.id}]\n{chunk.text}" for chunk in chunks
    )
    
    prompt = RESOLVE_PROMPT.format(
        question=question,
        sources=sources_str
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    result = json.loads(response.choices[0].message.content)
    
    return Ingredient(
        question=question,
        status=result["status"],
        answer=result.get("answer"),
        source_chunk_id=result.get("source_chunk_id")
    )


# =============================================================================
# MAIN: Build recipe for a seed
# =============================================================================

def build_recipe(
    seed: SelectedSeed,
    chunks: List[Chunk],
    row_instructions: str,
    column_schema: str,
    rag_top_k: int = 3,
    rag_min_score: float = 0.0
) -> Recipe:
    """
    Build a complete recipe for a seed.
    
    1. Analyze seed for completeness + sub-questions
    2. RAG search for each sub-question
    3. Resolve each sub-question from RAG results
    
    ============================================================================
    PRODUCTION IMPLEMENTATION
    ============================================================================
    
    FULL SIGNATURE:
    
        def build_recipe(
            seed: SelectedSeed,
            project_id: UUID,
            db_session: Session,
            row_instructions: str,
            column_schema: str,
            rag_top_k: int = 3,
            rag_min_score: float = 0.3
        ) -> Recipe:
    
    CHANGES:
    - Remove `chunks` parameter (query from DB instead)
    - Add `project_id` to scope RAG queries
    - Add `db_session` for pgvector access
    
    PARALLELIZATION:
    Process multiple seeds concurrently:
    
        async def build_recipes_batch(
            seeds: List[SelectedSeed],
            ...
        ) -> List[Recipe]:
            # Option A: Full parallel
            tasks = [build_recipe_async(seed, ...) for seed in seeds]
            return await asyncio.gather(*tasks)
            
            # Option B: Controlled concurrency
            semaphore = asyncio.Semaphore(10)  # Max 10 concurrent
            async def limited_build(seed):
                async with semaphore:
                    return await build_recipe_async(seed, ...)
            
            tasks = [limited_build(seed) for seed in seeds]
            return await asyncio.gather(*tasks)
    
    PERSISTENCE:
    Store recipes for job resume:
    
        # After building recipe
        db_session.add(RecipeModel(
            id=uuid4(),
            project_id=project_id,
            seed_id=seed.id,
            completeness=recipe.completeness,
            ingredients=jsonable_encoder(recipe.ingredients),
            status="ready"
        ))
        db_session.commit()
    
    ERROR HANDLING:
    
        try:
            recipe = build_recipe(seed, ...)
        except OpenAIError as e:
            # Rate limit, timeout, etc.
            mark_seed_for_retry(seed.id)
            raise
        except JSONDecodeError as e:
            # LLM returned invalid JSON
            log_malformed_response(seed.id, response)
            recipe = build_recipe_fallback(seed, ...)  # Simpler prompt
    
    ============================================================================
    """
    # Step 1: Analyze
    analysis = analyze_seed(seed, row_instructions, column_schema)
    completeness = analysis["completeness"]
    sub_questions = analysis["sub_questions"]
    
    # Step 2 & 3: RAG + Resolve for each question
    ingredients = []
    for question in sub_questions:
        relevant_chunks = rag_search(
            question, 
            chunks, 
            top_k=rag_top_k,
            min_score=rag_min_score
        )
        ingredient = resolve_question(question, relevant_chunks)
        ingredients.append(ingredient)
    
    return Recipe(
        seed_id=seed.id,
        seed_text=seed.text,
        assigned=seed.assigned,
        completeness=completeness,
        ingredients=ingredients
    )


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    row_instructions = """Each row is a multi-turn roleplay conversation between a user and an AI character. The conversation should feel natural and immersive. The AI character has a defined persona established in the system message."""
    
    column_schema = """conversation (array of message objects):
      - from: "system" | "human" | "gpt"
      - value: string (the message content)
    Structure:
    - First message is "system" defining the AI character/persona
    - Alternating "human" and "gpt" messages follow
    - Minimum 2 turns (4 messages after system)"""

    seed = SelectedSeed(
        id="seed_1",
        text="*His name is Damian, he's a dangerous criminal knows also as a mafia boss in the town. You just moved in the town and wanted to know your neighbors, it turned out your neighbor was Damian.*",
        assigned={"genre": "crime", "tone": "dark"},
        source_chunk_id="chunk_1"
    )
    
    print("Creating mock chunks with embeddings...")
    mock_chunks = [
        Chunk(
            id="chunk_1",
            text="Damian grew up in the streets of Brooklyn. His father was killed when he was 12, forcing him into the criminal underworld. He rose through the ranks through a combination of intelligence and ruthlessness.",
            embedding=compute_embedding("Damian grew up in the streets of Brooklyn...")
        ),
        Chunk(
            id="chunk_2", 
            text="The town is called Ravenswood, a small coastal city known for its foggy nights and old Victorian architecture. The mafia has controlled the docks for decades.",
            embedding=compute_embedding("The town is called Ravenswood...")
        ),
        Chunk(
            id="chunk_3",
            text="Damian's mansion sits on the hill overlooking the town. He rarely has visitors and is known to be suspicious of newcomers.",
            embedding=compute_embedding("Damian's mansion sits on the hill...")
        ),
    ]
    
    print("Building recipe...")
    recipe = build_recipe(
        seed=seed,
        chunks=mock_chunks,
        row_instructions=row_instructions,
        column_schema=column_schema,
        rag_top_k=3,
        rag_min_score=0.0
    )
    
    print("\n" + "="*50)
    print("RECIPE")
    print("="*50)
    print(f"Seed ID: {recipe.seed_id}")
    print(f"Completeness: {recipe.completeness}")
    print(f"Assigned: {recipe.assigned}")
    print(f"\nIngredients ({len(recipe.ingredients)}):")
    for ing in recipe.ingredients:
        print(f"\n  Q: {ing.question}")
        print(f"  Status: {ing.status}")
        print(f"  Answer: {ing.answer[:100]}..." if ing.answer and len(ing.answer) > 100 else f"  Answer: {ing.answer}")
        print(f"  Source: {ing.source_chunk_id}")