"""Extractive (template-based) answer generator.

Use this when you want a deterministic, fast, no-LLM answer that still surfaces
citations. Real Week-1 demo uses this so Milind has a working pipeline today;
swap in `Generator` (Hugging Face transformers) for production.

Logic:
- Concatenate the top-k chunks in order.
- Extract sentences that contain the most question keywords.
- Build a short answer paragraph with citations.
"""

from __future__ import annotations

import re

from src.retriever import RetrievedChunk


_SENTENCE_RE = re.compile(r"(?<=[.!?\n])\s+")



def _extract_company_names(text: str) -> list[str]:
    """Extract company names from job title lines.

    Matches patterns like:
      Sales Associate – Zudio (Tata Group), New Delhi, India -> [Zudio (Tata Group)]
      Sales Associate – Lifestyle (Lifestyle Retail), New Delhi -> [Lifestyle (Lifestyle Retail)]
    """
    companies: list[str] = []
    for m in re.finditer(r"[–\-]\s*([A-Z][a-zA-Z& ]+?(?:\([A-Za-z& ]+\))?)(?:,|$)", text):
        name = m.group(1).strip()
        if len(name) > 2 and name not in companies:
            companies.append(name)
    return companies

def _split_sentences(text: str) -> list[str]:
    # Split on newlines first (resumes use \n as paragraph breaks)
    text = text.strip()
    parts = re.split(r"\n", text)
    # Then split each part on sentence-ending punctuation
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        sub = _SENTENCE_RE.split(part)
        sentences.extend(s for s in sub if s.strip())
    return sentences if sentences else [text]


def _score_sentence(sentence: str, question_keywords: set[str]) -> int:
    s = sentence.lower()
    score = 0
    for kw in question_keywords:
        if kw in s:
            score += 2  # Full keyword match gets 2 points
        # Partial match for word stems
        for word in re.findall(r"\w+", s):
            if kw in word or word in kw:
                score += 1
                break
    return score


class ExtractiveGenerator:
    """Deterministic, no-LLM answer generator with citations."""

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    @staticmethod
    def _is_detail_request(question: str) -> bool:
        """Check if the question asks for a detailed answer."""
        q = question.lower()
        return any(kw in q for kw in ("detail", "explain", "elaborate",
                                        "tell me more", "describe", "more about"))

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        if not retrieved:
            return "I don't know based on the provided documents."

        detail = self._is_detail_request(question)
        # Concise by default (1-2 sentences); more detail on request.
        max_sentences = 5 if detail else self.max_sentences

        # Build keyword set (drop stopwords + tiny tokens).
        stop = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
            "to", "with", "what", "when", "where", "which", "who", "why", "how",
            "do", "does", "can", "could", "should", "would", "i", "you", "we",
            "they", "he", "she", "them", "us", "my", "your", "their",
            "this", "these", "those", "there", "then", "but", "not", "no",
        }
        keywords = {
            w.lower()
            for w in re.findall(r"\w+", question)
            if len(w) > 2 and w.lower() not in stop
        }
        # Always include key question terms for better matching
        for term in ("company", "companies", "work", "worked", "employ", "employed",
                     "experience", "retail", "fashion", "sales", "associate", "tata",
                     "zudio", "lifestyle"):
            if term in question.lower():
                keywords.add(term)
        if not keywords:
            keywords = {"main"}

        # First, look for job title patterns (e.g. "Sales Associate – Zudio")
        # This catches employment history even when the question doesn't name the company
        candidates: list[tuple[int, str, str]] = []  # (score, sentence, chunk_id)
        for rc in retrieved:
            for sent in _split_sentences(rc.chunk.text):
                if re.search(r"(Sales|Associate|Engineer|Manager|Intern|Developer|Consultant|Designer|Specialist)\s*[–\-]\s*", sent):
                    candidates.append((2, sent, rc.chunk.chunk_id))

        # Then pick best sentences by keyword score from top-k chunks
        for rc in retrieved:
            for sent in _split_sentences(rc.chunk.text):
                sc = _score_sentence(sent, keywords)
                if sc > 0:
                    if not any(c[1] == sent for c in candidates):
                        candidates.append((sc, sent, rc.chunk.chunk_id))

        # Fall back: take the first sentence of the top chunk if nothing matched.
        if not candidates:
            top = retrieved[0]
            first_sentence = _split_sentences(top.chunk.text)[0] or top.chunk.text[:200]
            return (
                f"{first_sentence.strip()}\n\n"
                f"Source: [{top.chunk.chunk_id}] (page {top.chunk.page_number})"
            )

        # Take the top-N by score, preserving order of appearance.
        candidates.sort(key=lambda t: (-t[0], retrieved[0].chunk.chunk_id))
        top = candidates[: max_sentences]

        if detail:
            # Detail mode: join full sentences with context.
            body = " ".join(s for _, s, _ in top).strip()
        else:
            # Concise mode: extract clean company names or key facts.
            body = " ".join(s for _, s, _ in top).strip()
            company_names = _extract_company_names(body)
            if company_names:
                body = " & ".join(company_names)

        # Build citation footer (unique chunk IDs only).
        seen: set[str] = set()
        citations: list[str] = []
        for rc in retrieved:
            cid = rc.chunk.chunk_id
            if cid in seen:
                continue
            seen.add(cid)
            citations.append(f"[{cid}] page {rc.chunk.page_number}")
        footer = " · ".join(citations)

        return f"{body}\n\nSources: {footer}"