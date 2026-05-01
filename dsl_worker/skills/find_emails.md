---
name: find_emails
description: Finding work emails for individuals
applies_to: [cell_agent]
triggers:
  - email
  - work email
  - business email
  - contact email
  - verified email
  - email_address
  - email address
---

## Finding someone's work email

**Cheapest-to-most-expensive ladder:**
1. **FullEnrich** (`fullenrich_enrich_contacts`) with name + company
   domain. Verified emails > guessed emails.
2. **Apollo** (`apollo_enrich_person`) as fallback.
3. **Pattern guess** ONLY if you have strong evidence of the org's email
   format (e.g. you've seen `firstname@company.com` confirmed for another
   employee at the same org). NEVER fabricate patterns and call them
   verified.

**Critical rules:**
- A confident null is better than a guessed email. The user can't audit
  a wrong email until it bounces — that's worse than empty.
- If FE returns "no_data" or "low_confidence", commit null. Don't fall
  through to guessing.
- Match the column format ("lowercase email or null", etc.) exactly. Strip
  trailing whitespace and normalize case before committing.

**Source signals worth chasing:**
- A LinkedIn URL or domain in another column → use it as input to
  enrichment APIs rather than re-searching.
- A company website → derive the email domain (about page, footer)
  before falling back to `@gmail.com` style guesses.
