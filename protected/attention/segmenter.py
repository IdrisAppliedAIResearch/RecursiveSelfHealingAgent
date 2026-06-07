import re
from dataclasses import dataclass

ABBREVIATIONS = {
    "e.g",
    "i.e",
    "vs",
    "Fig",
    "Eq",
    "al",
    "vol",
    "no",
}

ABBREV_PATTERN = re.compile(
    r'\b(?:e\.g\.|i\.e\.|vs\.|Fig\.|Eq\.|et al\.|vol\.|no\.)\b'
)


@dataclass
class Sentence:
    text: str
    label: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int


RESULTS_PATTERNS = [
    r'\b(showed?|demonstrated?|revealed?|found|observed|identified|detected)\b',
    r'\b(significantly|greater than|less than|more than|compared to|relative to)\b',
    r'\b(activation|deactivation|correlation|increase[sd]?|decrease[sd]?|reduction)\b',
    r'\b(p\s*[<>=]\s*0\.\d+|t\s*\(\d+\)|F\s*\(\d+|r\s*=\s*[-\d.])\b',
    r'\b(higher|lower|larger|smaller|stronger|weaker)\b.{0,40}\b(than|compared)\b',
    r'\b(bilateral|unilateral|left|right)\b.{0,30}\b(cortex|gyrus|sulcus|area|region)\b',
]

METHODS_PATTERNS = [
    r'\b(participants?|subjects?|volunteers?|patients?)\b.{0,20}\b(were|had|completed)\b',
    r'\b(fMRI|MRI|PET|EEG|MEG)\b.{0,30}\b(scanner|session|protocol|study)\b',
    r'\b(TR|TE|voxel|slice|mm|tesla|T)\b',
    r'\b(we used|study (examined|investigated|aimed|used)|designed to)\b',
    r'\b(informed consent|ethics|IRB|approved)\b',
    r'\b(\d+\s*(male|female|men|women|participants|subjects))\b',
]

_results_compiled = [re.compile(p, re.IGNORECASE) for p in RESULTS_PATTERNS]
_methods_compiled = [re.compile(p, re.IGNORECASE) for p in METHODS_PATTERNS]


def _split_sentences(abstract_text: str) -> list[tuple[str, int, int]]:
    boundary_positions = []
    i = 0
    while i < len(abstract_text) - 1:
        if abstract_text[i] == '.' and abstract_text[i + 1] == ' ':
            before = abstract_text[:i]
            last_dot_pos = before.rfind('.')
            if last_dot_pos >= 0:
                preceding_word_end = i - 1
                while preceding_word_end >= 0 and abstract_text[preceding_word_end].isalnum():
                    preceding_word_end -= 1
                preceding_word = abstract_text[preceding_word_end + 1:i].strip().rstrip('.')
                if preceding_word.lower() in ABBREVIATIONS:
                    i += 1
                    continue
            if last_dot_pos >= 0:
                word_after_last = abstract_text[last_dot_pos + 1:i].strip()
                if word_after_last and word_after_last[0].isupper():
                    i += 1
                    continue
            char_after_space = i + 2
            if char_after_space < len(abstract_text) and abstract_text[char_after_space].islower():
                last_space = abstract_text.rfind(' ', last_dot_pos, i)
                if last_space >= 0:
                    preceding_token = abstract_text[last_space + 1:i].strip()
                    if preceding_token.lower() in ABBREVIATIONS:
                        i += 1
                        continue
            boundary_positions.append(i + 1)
        i += 1

    sentences = []
    start = 0
    for end_pos in boundary_positions:
        text = abstract_text[start:end_pos].strip()
        if text:
            sentences.append((text, start, end_pos))
        start = end_pos
    trailing = abstract_text[start:].strip()
    if trailing:
        sentences.append((trailing, start, len(abstract_text)))

    return sentences


def _classify_sentence(sentence_text: str) -> str:
    results_hits = sum(
        1 for p in _results_compiled if p.search(sentence_text)
    )
    methods_hits = sum(
        1 for p in _methods_compiled if p.search(sentence_text)
    )

    if results_hits > methods_hits:
        return "RESULTS"
    if methods_hits > results_hits:
        return "METHODS"
    if results_hits == methods_hits and results_hits > 0:
        return "RESULTS"
    if results_hits > 0:
        return "RESULTS"
    if methods_hits > 0:
        return "METHODS"
    return "BACKGROUND"


def segment_abstract(abstract_text: str) -> list[Sentence]:
    raw_sentences = _split_sentences(abstract_text)
    sentences = []
    for text, char_start, char_end in raw_sentences:
        label = _classify_sentence(text)
        sentences.append(
            Sentence(
                text=text,
                label=label,
                char_start=char_start,
                char_end=char_end,
                token_start=0,
                token_end=0,
            )
        )
    return sentences


def align_tokens(sentences: list[Sentence], tokenizer, abstract_text: str) -> None:
    encoding = tokenizer(abstract_text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    token_to_char = []
    for (start, end) in offsets:
        token_to_char.append(start)

    for sent in sentences:
        sent_char_start = sent.char_start
        sent_char_end = sent.char_end
        first_token = 0
        last_token = 0
        for idx, (t_start, t_end) in enumerate(offsets):
            if t_start >= sent_char_end:
                break
            if t_end > sent_char_start:
                if first_token == 0 and t_start >= sent_char_start:
                    first_token = idx
                last_token = idx + 1

        if first_token == 0:
            for idx, (t_start, t_end) in enumerate(offsets):
                if t_start >= sent_char_start:
                    first_token = idx
                    break
            else:
                first_token = 0

        sent.token_start = first_token
        sent.token_end = last_token
