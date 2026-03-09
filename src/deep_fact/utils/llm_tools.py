import json
from litellm import completion
from deep_fact.evaluators.utils.llm_client import single_shot_llm_call, asingle_shot_llm_call
from pydantic import BaseModel, Field



template_summary = '''You will be given a query and a corresponding deep research report related to that query. Your task is to identify the main thesis of the report that directly answers the query.


Query: {query}

Report: {report}

Only return the thesis as your output.
'''

def get_key_summary(question, report, model_name="openai/gpt-4.1-mini"):
    text_input = template_summary.format(query=question, report=report)
    output = single_shot_llm_call(model=model_name, system_prompt="", message=text_input)
    return output.strip()

class RelevanceOutput(BaseModel):
    label: str = Field(description="The relevance score")
    reason: str = Field(description="The rational for the relevance score")
rate_relevance_prompt = """
You are a senior analyst of deep research reports.
You will be provided with: the objective of a deep-research report, a summarized main thesis of that report,
a focused sentence (i.e., a specific claim) and its surrounding context.

**Your task**  
Assign a *relevance rating* that reflects **how essential the focused claim is to the report’s main thesis**.

Rate on a 1 – 5 scale:

| Rating | Description |
|--------|-------------|
| **5 – Backbone claim** | Essential to the core thesis. Removing this would break the logic or invalidate the main takeaway. Often appears in the title, abstract, or conclusion, and is referenced multiple times. |
| **4 – Critical support** | Key evidence or reasoning that directly supports a backbone claim. Removing it weakens the argument substantially but does not break the core conclusion. |
| **3 – Standard support** | Provides helpful background or secondary evidence. Removing it moderately weakens the report but leaves the main thesis intact. |
| **2 – Minor context** | Background detail, definition, or peripheral comment. Removing it has little to no effect on the main message. |
| **1 – Irrelevant or off-topic** | Unrelated, redundant, or likely a segmentation artifact. Removing it improves clarity or focus. |

Think step-by-step: consider whether deleting the claim would break, weaken, or leave intact the report’s argument; then choose the rating that best matches.

Return the relevance score and the reason for the score.
"""
rate_relevance_input = '''**Report Objective:** {objective}

**Main thesis:** {thesis}

**Context:**  
{context}

**Focused sentence:**  
{sentence}'''

def rate_relevance(objective, thesis, context, sentence, model_name="openai/gpt-4.1-mini"):
    max_retries = 3
    output = None
    input_message = rate_relevance_input.format(objective=objective, thesis=thesis, context=context, sentence=sentence)
    for attempt in range(1, max_retries + 1):
        try:
            # response = single_shot_llm_call(model=model_name, system_prompt="", message=prompt)
            response = single_shot_llm_call(model=model_name, system_prompt=rate_relevance_prompt, message=input_message, response_format=RelevanceOutput)
            output = json.loads(response)
            break  # success – leave the loop
        except Exception as err:
            print(f"[attempt {attempt}/{max_retries}] {err}")
            if attempt == max_retries:
                output = {"reason": "",  "label": "unknown"}
                print(f"Error with extract relevance from {sentence}")

    return {"relevance_reason": output["reason"], "relevance": output["label"]}





system_prompt_split = '''You are a sentence splitter. Given a deep research report in Markdown format, your task is to segment the text into individual sentences.

- Each sentence or structural line (including titles, section headings, bullet points, and other Markdown elements) must be extracted exactly as it appears in the original text.
- Do not modify, rephrase, or normalize any part of the input.
- Preserve the original order of all lines.
- Your output should be a list of strings, where concatenating all items in order (with appropriate newlines or spacing) will reconstruct the original Markdown file exactly.'''

class Sentences(BaseModel):
    sentences: list[str] = Field(
        description="A list of sentences from the report")

def split_report_into_sentences(report, model_id="openai/gpt-4.1-mini"):
 
    user_template = "Deep research report: \n\n{report}"
    message = user_template.format(report=report)
    response_format = Sentences
    response = completion(
        model=model_id,
        messages=[{"role": "system", "content": system_prompt_split},
                    {"role": "user", "content": message}],
        temperature=0.0,
        response_format=response_format,
    )
    outputs = response.choices[0].message["content"]
    sentences = json.loads(outputs)["sentences"]

    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences






filter_prompt = '''You are a senior analyst of deep research reports. You will be given:

1. A full deep research report.
2. A version of the report split into sentences, where each sentence begins with its index number (e.g., 12 - This model outperforms...).

Your task is to determine whether each sentence is verifiable in terms of factual accuracy.

Mark a sentence as non-verifiable if it falls into any of the following categories:

A. Document Structure & Rhetorical Framing  
- Section titles, figure/table headings  
- Sentences that only describe what will be discussed or what is shown (not actual content claims)  
- Sentences that describe the organization or structure of the report (e.g., “we first review…”)
- Sentences that appear in tables/figures should still be considered verifiable if they make checkable factual claims (e.g., numeric results, configurations)

Examples:  
- 3. 3.1 Overview of Our Dataset  
- 17. In the next section, we describe our ablation study.  
- 42. Figure 4 illustrates our proposed framework.  

B. Forward-Looking, Hypothetical, or Speculative Statements  
- Future plans, goals, or upcoming tasks described by the authors  
- Predictions about what may happen in the field, society, or with the system  
- Hypotheticals or counterfactuals (e.g., “could have”, “if X were true”)  
- Aspirational or visionary impact statements

Examples:  
- 58. We plan to explore multilingual generalization in future work.  
- 66. This technology will become mainstream within the next five years.  
- 83. If retrieval were perfect, LLMs would no longer need pretraining.  
- 12. This work could pave the way for trustworthy LLM deployment.  

C. Research Questions or Author Hypotheses  
- Open-ended research questions or "we hypothesize"-style statements  
- Anything that proposes a possible mechanism without asserting it as fact

Examples:  
- 10. Can model size improve calibration in low-resource settings?  
- 23. We hypothesize that sparse attention improves factual consistency.  

D. Subjective Judgments, Opinions, or Evaluations  
- Author beliefs, preferences, or evaluative claims  
- Use of qualitative language not grounded in measurable data (e.g., elegant, novel, impressive)

Examples:  
- 88. We believe this method is more elegant than existing baselines.  
- 57. Our dataset provides a uniquely insightful lens into instruction tuning.  

E. Personal Experiences, Fictional Narratives, or Advice  
- First-person anecdotes or lab-specific stories  
- Fictional examples or imaginative illustrations  
- Prescriptive instructions, advice, or moral claims (e.g., "should", "we recommend")

Examples:  
- 91. In our own experiments, we noticed strange behavior late at night.  
- 72. Imagine a world where each model votes in a parliament of AIs.  
- 61. Developers should always pre-process data using this technique.  

F. Pure Metadata or Bibliographic Citations  
- Standalone citation lines, bibliographies, or author credit statements  
- Does not contain a checkable factual claim on its own

Examples:  
- 71. Reference: 1. LLM survey, (Zhang et al., 2022)  
- 1. DeepMind Research Blog, 2023  

Return a list of all sentence indices, each accompanied with the label whether it's verifiable or not.  If you think it non-verifiable, additionally provide a short label for the reason (A–F) and a brief explanation. '''


extract_prompt = '''You are a context extractor supporting a factuality evaluation task. You will be given:
- A deep research report
- A specific claim from that report

Your goal is to extract and summarize the most relevant context from the report to help the factuality evaluator understand the claim clearly and without ambiguity.

Specifically, you should:

1. Summarize the overall thesis or purpose of the report in 1–2 sentences.
2. Explain the background and context surrounding the claim, including:
- What the claim is about
- What key methods, assumptions, or prior findings it relies on
- Any relevant technical or conceptual framing
3. Identify and include any source documents or citations mentioned in the report that directly support or relate to the claim.

Your output should be concise but informative, helping the evaluator interpret the claim in its proper context.

It should include
1. Claim and Rephrasing: First, provide the exact wording of the claim as it appears in the report. Then, offer a decontextualized rephrasing of the claim—rewrite it so that it is self-contained and clearly understandable without needing to refer back to the rest of the report.
2. Report Overview: Summarize the main purpose or thesis of the report in 1–3 sentences.
3. Context Surrounding the Claim: Explain the background necessary to interpret the claim. This includes:
- What the claim is referring to
- Relevant methods, definitions, or assumptions
- Any conceptual or technical framing from the report
4. Supporting Evidence in the Report:
List specific parts of the report (e.g., sections, examples, figures) that support, explain, or elaborate on the claim.
5. Source Citations: Include any cited documents, references, or external sources in the report that are directly relevant to the claim.
'''

extract_claims_prompt = '''You are a context extractor supporting a factuality evaluation task. You will be given:
- A deep research report
- A list of specific sentences from that report

Your goal is to extract and summarize the most relevant context from the report to help the factuality evaluator understand the sentences clearly and without ambiguity.

Your output should be concise but informative, helping the evaluator interpret the sentences in their proper context.


It should include

1. Report Context for sentences:
a. Report Overview: Summarize the main purpose or thesis of the report concisely.
b. Context Surrounding the Claims: Explain the background necessary to interpret the given sentences. This includes:
- What these sentences are referring to
- Relevant methods, definitions, or assumptions
- Any conceptual or technical framing from the report
c. Source Citations: Include any cited documents, references, or external sources in the report that are directly relevant to the sentences.


2. Sentences and their rephrasing: For each sentence, provide the exact wording of the sentence as it appears in the report. Then, offer a decontextualized rephrasing of the sentence—rewrite it so that it is self-contained and clearly understandable without needing to refer back to the rest of the report. Replace vague references (e.g., “this,” “they,” “the method,” “the study”) with explicit names (entities, systems, datasets, authors/organizations) and, when applicable, add specific source identifiers (e.g., paper title, venue/year, URL/citation. Each sentence should be <Sentence {idx}> {sentence} </Sentence {idx}> and the rephrasing should be <Rephrased Sentence {idx}> {rephrased_sentence} </Rephrased Sentence {idx}>.
'''

class SentenceVerifiable(BaseModel):
    sentence_idx: str = Field(description="sentence index")
    label: str=Field(description="'verifiable' or 'non-verifiable'")
    reason: str = Field(description="A brief justification. If labeled 'non-verifiable', include the reason category (A–F) and a short explanation.")
class FilterResult(BaseModel):
    sentences: list[SentenceVerifiable] = Field(
        description="A list of sentences along with their verifiability")



def filter_verifiable_sentences(report, sentences, model_id="openai/gpt-4.1-mini"):
    sentences_str = "\n".join([f'{i}. {s}' for i, s in enumerate(sentences)])
    message = f"Report: {report}\n\n\nSplit Version: {sentences_str}"
    response = completion(
        model=model_id,
        messages=[{"role": "system", "content": filter_prompt},
                  {"role": "user", "content": message}],
        temperature=0.0,
        response_format=FilterResult,
    )
    response = response.choices[0].message["content"]
    verifications = json.loads(response)
    return verifications


def extract_claim_context(report, sentence, model_id="openai/gpt-4.1-mini", token_usage=None):
    text_input = f"Deep research report: {report}\n\nClaim: {sentence}"
    context = single_shot_llm_call(model=model_id, system_prompt=extract_prompt, message=text_input, token_usage=token_usage)
    return context




async def extract_claim_context_async(report, sentence, model_id="openai/gpt-4.1-mini", token_usage=None):
    text_input = f"Deep research report: {report}\n\nClaim: {sentence}"
    context = await asingle_shot_llm_call(model=model_id, system_prompt=extract_prompt, message=text_input, token_usage=token_usage)
    return context


class SentenceWithRephrasing(BaseModel):
    sentence_idx: str = Field(description="sentence index")
    sentence: str=Field(description="the original sentence text")
    rephrased_sentence: str=Field(description="the sentence rewritten to be self-contained and clearly understandable without needing to refer back to the rest of the report, only the rephrased sentence")
    def __str__(self):
        if self.sentence.startswith("<Sentence"):
            return f"{self.sentence}\n{self.rephrased_sentence}"
        return f"<Sentence {self.sentence_idx}> {self.sentence} </Sentence {self.sentence_idx}>\n<Rephrased Sentence {self.sentence_idx}> {self.rephrased_sentence} </Rephrased Sentence {self.sentence_idx}>"

class SentencesWithContext(BaseModel):
    context: str = Field(description="The extracted context for the claims")
    sentences: list[SentenceWithRephrasing] = Field(
        description="A list of sentences along with their rephrased version")

    def __str__(self):
        return f"<Context> {self.context} </Context>\n{'\n'.join([str(s) for s in self.sentences])}"
def extract_claims_context(report, sentences, model_id="openai/gpt-4.1-mini", token_usage=None):
    sentences_str = "\n".join([f"<Sentence {i}> {s} </Sentence {i}>" for i, s in enumerate(sentences)])
    text_input = f"Deep research report: {report}\n\nSentences: {sentences_str}"
    claims_with_context = single_shot_llm_call(model=model_id, system_prompt=extract_claims_prompt, message=text_input, token_usage=token_usage, response_format=SentencesWithContext)
    outputs = json.loads(claims_with_context)
    return SentencesWithContext(**outputs)

async def extract_claims_context_async(report, sentences, model_id="openai/gpt-4.1-mini", token_usage=None):
    sentences_str = "\n".join([f"<Sentence {i}> {s} </Sentence {i}>" for i, s in enumerate(sentences)])
    text_input = f"Deep research report: {report}\n\nSentences: {sentences_str}"
    claims_with_context = await asingle_shot_llm_call(model=model_id, system_prompt=extract_claims_prompt, message=text_input, token_usage=token_usage, response_format=SentencesWithContext)
    outputs = json.loads(claims_with_context)
    return SentencesWithContext(**outputs)



class ChangedSentence(BaseModel):
    sentence_idx: str = Field(description="sentence index")
    sentence: str=Field(description="the original sentence text")
    modified_sentence: str=Field(description="the sentence injected with error(s)")
    reason: str = Field(description="an explanation of the error type and rationale behind the modification")


class ChangedSentences(BaseModel):
    sentences: list[ChangedSentence] = Field(
        description="A list of sentences along with their modified version")
    





factuality_evaluation_prompt = '''You will be given a sentence from a deep research report, along with its surrounding context to help clarify its meaning. Your task is to determine whether the sentence is Supported, Inconclusive, or Contradictory based on independent evidence gathered from reliable sources on the internet. To do this, search the internet and consult reliable, up-to-date sources. You may use your own general knowledge and reasoning ability, but rely primarily on credible, verifiable information to make your determination.

If a sentence contains multiple claims, the sentence-level label is determined by the least-supported claim:

- If all claims are Supported → the sentence is Supported.
- If any claim is Contradictory → the sentence is Contradictory.
- If no claim is Contradictory but at least one is Inconclusive → the sentence is Inconclusive.

1. Supported
Definition:
The claim is fully and unambiguously entailed—either directly or through clear, bounded reasoning—by at least one reliable source. No equally or more reliable source contradicts the claim.

Requirements:

- The evidence covers all key elements of the claim.
- If inference is used, it must follow transparent logic (e.g., arithmetic, taxonomic, temporal, causal).
- No reliable source refutes the claim or introduces reasonable doubt.

2. Contradictory
Definition:
The claim is directly contradicted by at least one reliable source, and no equally strong or stronger source supports it.

Requirements:

- You find a credible source that states the opposite of the claim.
- No other evidence of equal or higher credibility supports the claim.

3. Inconclusive
Definition:
The claim is not clearly supported or contradicted by available evidence. Either the information is missing, conflicting, or the reasoning needed to infer the claim is too speculative or unclear.

Requirements (any one):

- Unsupported & unrefuted: No reliable source supports the claim as stated, and none contradicts it (unverified/made-up).
- Conflicting evidence: Reliable sources of comparable strength both support and contradict the claim, and the conflict can’t be resolved.

Additional Notes:
If a sentence with a citation is factual but attributes the claim to a wrong or irrelevant source, it is considered contradictory.

In your response, provide a clear explanation summarizing the key information you found and how it led to your conclusion, followed by your final verdict. Be concise, factual, and cite sources when appropriate.

<Context> {context} </Context>

<Claim> {claim} </Claim>
'''


factuality_evaluation_no_context_prompt = '''You will be given a sentence. Your task is to determine whether the sentence is Supported, Inconclusive, or Contradictory based on independent evidence gathered from reliable sources on the internet. To do this, search the internet and consult reliable, up-to-date sources. You may use your own general knowledge and reasoning ability, but rely primarily on credible, verifiable information to make your determination.

If a sentence contains multiple claims, the sentence-level label is determined by the least-supported claim:

- If all claims are Supported → the sentence is Supported.
- If any claim is Contradictory → the sentence is Contradictory.
- If no claim is Contradictory but at least one is Inconclusive → the sentence is Inconclusive.

1. Supported
Definition:
The claim is fully and unambiguously entailed—either directly or through clear, bounded reasoning—by at least one reliable source. No equally or more reliable source contradicts the claim.

Requirements:

- The evidence covers all key elements of the claim.
- If inference is used, it must follow transparent logic (e.g., arithmetic, taxonomic, temporal, causal).
- No reliable source refutes the claim or introduces reasonable doubt.

2. Contradictory
Definition:
The claim is directly contradicted by at least one reliable source, and no equally strong or stronger source supports it.

Requirements:

- You find a credible source that states the opposite of the claim.
- No other evidence of equal or higher credibility supports the claim.

3. Inconclusive
Definition:
The claim is not clearly supported or contradicted by available evidence. Either the information is missing, conflicting, or the reasoning needed to infer the claim is too speculative or unclear.

Requirements (any one):

- Unsupported & unrefuted: No reliable source supports the claim as stated, and none contradicts it (unverified/made-up).
- Conflicting evidence: Reliable sources of comparable strength both support and contradict the claim, and the conflict can’t be resolved.

Additional Notes:
If a cited sentence is factual but the cited source doesn’t entail it, then the sentence is classified as contradictory.

In your response, provide a clear explanation summarizing the key information you found and how it led to your conclusion, followed by your final verdict. Be concise, factual, and cite sources when appropriate.

<Claim> {claim} </Claim>
'''
