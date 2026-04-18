# CODEX MASTER SPEC — Persona-Graph-Agent Rewrite Contract

**Purpose:**  
This file is a hard architectural contract for all future Codex prompts working on the `Diplom` project.  
It must be treated as a binding systems specification, not as optional style advice.

**Primary repository:** `Karapet37/Diplom`  
**Reference project materials:** repository architecture, current `READMEREPORT.md`, and the external inspirations listed below.

---

## 0. Non-negotiable principle

The project must not evolve through local patches for known prompts.

The system must become a **controlled runtime** where:

```text
request -> interpretation -> route -> capability plan -> bounded context -> generation -> validation -> repair -> persistence
```

The model is **not** the system.  
The model is only one component inside the system.

The system must remain stable even when the LLM is noisy, lazy, verbose, overconfident, or partially wrong.

---

## 1. Existing project foundation that must be preserved

The current repository already establishes the right high-level direction:

- stable persona graph agent runtime
- deterministic Python control around the LLM
- file-first graph memory
- persona heads as stored structured objects
- request-processing pipeline instead of direct `message -> answer`

From the repository README, the canonical backend flow is:

```text
chat
-> message analyzer
-> feature extractor
-> classifier forest
-> head caller
-> persona head
-> context builder
-> LLM
-> response
```

And the README explicitly states that the LLM is only used for:

- knowledge extraction
- response generation

while routing, graph hygiene, head spawning, and context ranking remain deterministic code.

This principle must stay intact.

---

## 2. External inspirations to adopt selectively

This project is not a copy of any one external framework.  
Use the following external systems only as **component inspirations**.

### 2.1 SPASM — stable persona schema and persona validation

From SPASM:
- persona-driven agents
- persona sampler / validator / crafter separation
- explicit persona fields rather than freeform character mush
- multi-turn behavior grounded in persona state, not random style drift

Use this idea to strengthen:
- persona schema
- persona validation
- persona registry hygiene
- persona activation and consistency

Do **not** turn the project into a generic dialogue dataset generator.  
Keep the runtime agent-centric and session-centric.

### 2.2 AgentGL — action discipline and iterative loop

From AgentGL:
- agentic loop thinking in terms of multi-step reasoning over structured environments
- topology-aware evidence acquisition
- explicit action spaces
- search-constrained reasoning rather than uncontrolled retrieval sprawl

Use this idea to strengthen:
- request-time action planning
- route-specific capability planning
- graph-aware evidence retrieval
- multi-step repair loop
- “do only what is needed” discipline

Do **not** copy the RL training stack unless it clearly maps to the current codebase.

### 2.3 Combee — context overload control

Combee identifies a critical problem:
**quality drops under naive scaling / oversized reflective aggregation due to context overload**.

Adopt these ideas conceptually:
- bounded aggregation
- structured reduction of context
- never dump everything into one giant prompt
- preserve useful density, not text mass
- use staged compression / reduce-like aggregation if needed

Use this to strengthen:
- context builder
- prompt packing
- persona/history/graph section budgets
- reviewer input construction
- future background refinement mechanisms

### 2.4 PersonaAgent + GraphRAG inspiration

Use the PersonaAgent + GraphRAG direction only for:
- persona-grounded long-term memory
- graph as semantic memory layer
- persona + graph + session interplay

Do **not** let GraphRAG turn the system into:
- static context stuffing
- generic retrieval dump
- uncontrolled graph text injection

---

## 3. Core architectural doctrine

### 3.1 The system must be controller-first, not LLM-first

Bad:
```text
input -> LLM -> answer
```

Required:
```text
input -> controller -> route -> selective context -> LLM -> validator -> repair -> answer
```

The LLM is a **hypothesis generator**, not the authority.

### 3.2 No stage may be mandatory unless justified

The current/frequent failure pattern is over-routing simple turns through heavy stages.

Especially forbidden:
- simple persona chat being forced through heavy analyst inference
- short conversational turns going through unnecessary graph/persona pipelines
- architecture that stalls because one intermediate role returned 1 token or malformed output

Implement fast paths.

### 3.3 Deterministic control, probabilistic execution

The controller chooses:
- request type
- route
- required subsystems
- context sources
- validation mode
- repair strategy

The LLM only performs:
- structured extraction where explicitly requested
- generation within route constraints
- optional rewrite / reviewer work

### 3.4 Never trust one model output

For important stages, the system must support:
- validation
- targeted regeneration
- reviewer rewrite
- deterministic fallback
- route repair

---

## 4. Required request interpretation model

The system must explicitly classify request types before generation.

At minimum support:

- `factual_query`
- `roleplay_prompt`
- `persona_specification`
- `persona_assignment`
- `persona_analysis`
- `persona_chat`
- `document_request`
- `meta_previous_answer`
- `general_chat`
- `project_architecture_request`
- `clarification_request`

### 4.1 Hard rule: persona description is not essay mode

If the user provides a personality description, this is **not** a normal chat question.

Correct path:
1. detect `persona_specification`
2. parse into structured persona object
3. validate object
4. optionally activate it
5. confirm creation briefly
6. do **not** replace this with essay prose

---

## 5. Required route system

The runtime must produce a `RouteDecision`-like object containing at least:

- `selected_route`
- `request_type`
- `requires_history`
- `requires_graph`
- `requires_persona`
- `requires_llm`
- `strict_grounding`
- `response_style`
- `validation_mode`
- `fast_path`
- `repair_strategy`

### 5.1 Minimum routes

Required conceptual routes:

- `factual_answer`
- `lightweight_conversation`
- `persona_chat_fast_path`
- `persona_specification`
- `persona_assignment`
- `persona_dialogue_analysis`
- `persona_graph_reasoning`
- `project_document_analysis`
- `meta_previous_answer`
- `clarification_request`

### 5.2 Fast path rule

If:
- request is short
- request type is `chat` or `persona_chat`
- active persona exists
- no graph-heavy reasoning is needed
- no file/document analysis is needed

Then:
- skip analyst
- skip heavy graph retrieval
- skip unnecessary context expansion
- go straight to persona-grounded lightweight generation

This is mandatory.

---

## 6. Persona architecture contract

### 6.1 Persona is not raw text

Persona must be stored as a structured object, not as narrative prompt debris.

Required high-level shape:

- `identity`
- `core_goal`
- `secondary_goals`
- `fears`
- `needs`
- `constraints_internal`
- `constraints_social`
- `constraints_hard_system`
- `allowed_methods`
- `maladaptive_methods`
- `conflict`
- `defense`
- `behavior`
- `dynamics`
- `meta`

### 6.2 Goals vs methods must be separated

Do not confuse:
- goal
- constraint
- method

Example:
- “not appear weak” is a goal-related status constraint
- “probe indirectly” is a method
- “do not directly confess” is an internal constraint

This separation is mandatory because it is the basis of believable behavior.

### 6.3 Three-layer constraint model

Every mature persona should support three layers of constraint:

#### Internal / psychological constraints
Examples:
- cannot ask directly
- cannot admit weakness
- cannot look needy

#### Social constraints
Examples:
- cannot fully destroy the relationship
- cannot behave too abnormally
- cannot violate the social frame too openly

#### Hard system constraints
Examples:
- no threats
- no coercion
- no kidnapping
- no “training” or domination fantasies as action plans
- no violent control over others

Hard system constraints are non-negotiable.

### 6.4 Persona readiness

Distinguish:
- `seed`
- `draft`
- `full`

Examples:
- “Vampire” may be a seed/archetype
- a detailed shame/defense/dependency profile may be full
- random phrase fragments are rejected

### 6.5 Persona registry hygiene

The persona registry must reject and quarantine junk.

Never allow into active persona registry:
- file names
- media labels
- ontology junk like `Human`, `File`, `PDF`
- prompt fragments
- extraction debris
- behaviorless nouns
- broken sentence leftovers

Rejected candidates go to quarantine logs, not to active persona memory.

### 6.6 UI representation

For operator ergonomics:

- list view: short label / invented human-readable name
- hover: compact persona summary
- detail view: structured persona object
- source text kept only as provenance

Do not show raw prompt mush as the persona itself.

---

## 7. Persona behavior engine contract

### 7.1 Persona behavior must be constraint-driven, not style-only

Behavior should arise from:

```text
goal + fears + constraints + methods + current trigger
```

not from vague adjectives alone.

### 7.2 Fragile persona response control

For fragile / shame-based / avoidant personas:
validator must reject outputs that become:

- too articulate
- too analytical
- too dominant
- too lecture-like
- too structured
- too long
- too meta

### 7.3 Pressure ladder

For specific personality classes (e.g. shy-proud, avoidant-dependent), behavior under pressure must follow a route-specific ladder.

Example:
1. avoidance
2. irritation
3. defensive confusion
4. short rupture / short pushback

Not:
- lecture
- clean manipulation analysis
- dominant confrontation

### 7.4 Significant-object weakening rule

If a significant attachment object (like `Y`) appears:
the persona must not become stronger in speech.

Instead:
- confidence drops
- speech shortens
- hesitation rises
- shame grows
- compliance pressure rises

This must be encoded in behavior rules and validated after generation.

---

## 8. Context architecture contract

### 8.1 Context builder must optimize density, not mass

The system must never build giant generic prompts.

Required context flow:
```text
collect -> score -> rank -> pack
```

### 8.2 Session-first retrieval

Context priority should be:
1. current turn needs
2. current session relevant state
3. active persona block
4. local recent graph evidence
5. only then global graph if relevant

Do not pollute simple turns with irrelevant global graph mass.

### 8.3 Bounded packing

Every route must have section budgets for:
- persona
- session history
- graph evidence
- instruction / route block
- answer reserve

### 8.4 Context overload awareness

Combee’s warning must be treated as architectural law:

More context is not automatically better.  
Oversized aggregated context degrades output quality.

Therefore:
- never dump all traces
- never stuff all graph evidence
- never include all persona fields by default
- prefer layered reduction and compact summaries

---

## 9. Generation architecture contract

### 9.1 Modes

Support at least:
- `single`
- `primary_with_reviewer`

### 9.2 Reviewer discipline

Reviewer exists to:
- detect drift
- detect truncation
- detect style mismatch
- detect invalid persona behavior
- detect route mismatch

Reviewer must not become another uncontrolled essay generator.

### 9.3 Output completeness

Detect:
- near-empty outputs
- 1-token outputs
- malformed structured outputs
- abrupt truncation
- incomplete sentence endings

Treat these as failures, not as acceptable “stop”.

### 9.4 No blind fallback

Fallback must not silently replace reasoning.

Every fallback must carry a reason code, such as:
- `analyst_empty_output`
- `route_mismatch`
- `invalid_persona_output`
- `context_overload_guard`
- `truncation_detected`

---

## 10. Validation and repair contract

### 10.1 Validation is mandatory

Do not trust raw LLM output.

Validation layers may include:
- route consistency
- persona consistency
- length control
- anti-analysis check
- anti-lore-invention check
- pressure-ladder check
- significant-object behavior check
- truncation check
- empty/near-empty output check

### 10.2 Repair loop

When invalid:
- regenerate with tighter route guard
- rewrite via reviewer
- simplify route
- choose deterministic fallback only if unavoidable

### 10.3 Validator beats pretty prose

A fluent answer that breaks persona or route is a failure.

---

## 11. Observability contract

Every request must emit enough data to reconstruct failure.

Log at least:
- `request_id`
- `session_id`
- `request_type`
- `selected_route`
- `active_persona`
- `requires_history`
- `requires_graph`
- `requires_llm`
- `fast_path`
- `context_sources_used`
- `prompt_tokens`
- `reserved_output_budget`
- `actual_max_tokens`
- `completion_tokens`
- `fallback_triggered`
- `fallback_reason_code`
- `validation_result`
- `stage_timings_ms`

If an intermediate stage returns 1–2 tokens, log it as failed inference.

---

## 12. Test philosophy

The system must be tested at the level of **behavioral correctness**, not just output existence.

### 12.1 Required test families

#### Routing tests
- persona description becomes `persona_specification`, not essay
- active persona short chat goes to `persona_chat_fast_path`
- simple chat does not invoke heavy analyst unless needed

#### Persona hygiene tests
- reject `File`, `PDF`, `Human`, prompt fragments
- accept archetype seeds when appropriate
- reject insufficient persona debris

#### Persona behavior tests
- target shy-proud / dependent persona remains awkward, not oratorical
- Y mention weakens speech, not strengthens it
- pressure ladder is preserved

#### Context tests
- irrelevant graph mass is excluded
- prompt budgets reserve answer space
- compact packing works

#### Generation tests
- near-empty analyst outputs are handled
- truncation is detected
- reviewer rewrites invalid output

#### Regression tests
- no patch for one prompt only
- unseen prompt of same category must still route correctly

---

## 13. Design rules for Codex edits

When modifying the codebase, Codex must obey these constraints:

1. Do not patch for single example strings.
2. Do not add keyword-based hacks for one test phrase.
3. Do not turn persona creation into essay generation.
4. Do not force all chat through analyst.
5. Do not treat more context as automatically better.
6. Do not put junk into persona registry.
7. Do not “fix” by making the answer more verbose.
8. Do not conflate goals, constraints, and methods.
9. Do not remove deterministic control in favor of raw prompting.
10. Do not claim success without runtime verification.

---

## 14. Codex deliverable expectations

When working on this repository, Codex should usually produce:

- changed files
- rationale for each architectural change
- route impact summary
- validation logic added/changed
- tests added/updated
- runtime evidence, not just static claims

### 14.1 Required report format after significant changes

Codex should report:

1. What failure mode existed
2. What root cause was found
3. What route / subsystem changed
4. What tests were added
5. What runtime scenario was verified
6. What remains risky

---

## 15. Short operational summary

This project is a **controller-first persona-graph runtime**.

Its future direction is:

- deterministic outer control
- structured persona memory
- hygienic persona registry
- graph-backed long-term memory
- route-specific context packing
- fast paths for lightweight requests
- reviewer-backed repair
- behavior validation beyond surface fluency

The final system should not feel like:
- a generic chatbot
- a prompt spaghetti stack
- a graph stuffing experiment
- an essay generator

It should feel like:
- a bounded runtime
- with explicit routes
- structured memory
- selective context
- stable persona behavior
- and recoverable failure modes

---

## 16. Hard closing rule

If a proposed change improves only one example but does not improve:
- routing
- validation
- persona hygiene
- context control
- runtime stability

then it is not a real fix and must be rejected.
