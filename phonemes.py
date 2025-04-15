import nltk
# nltk.download('cmudict')
from nltk.corpus import cmudict
from collections import defaultdict
from typing import List, Tuple, Dict, Sequence

def remove_stress(phoneme: str) -> str:
    return ''.join([c for c in phoneme if not c.isdigit()])

def load_cmu() -> List[Tuple[str, List[str]]]:
    cmu_raw = cmudict.dict()
    simplified_entries = []
    for word, pron_list in cmu_raw.items():
        for pron in pron_list:
            simplified_entries.append((word.upper(), [remove_stress(p) for p in pron]))
    return simplified_entries

def preprocess_pronunciation_dict(pron_dict: List[Tuple[str, List[str]]]) -> Dict[Tuple[str, ...], List[str]]:
    phoneme_to_words = defaultdict(list)
    for word, phonemes in pron_dict:
        phoneme_to_words[tuple(phonemes)].append(word)
    return phoneme_to_words

def find_word_combos_with_pronunciation(phonemes: Sequence[str], phoneme_to_words: Dict[Tuple[str, ...], List[str]]) -> Sequence[Sequence[str]]:
    memo = {}

    def backtrack(start: int) -> List[List[str]]:
        if start == len(phonemes):
            return [[]]
        if start in memo:
            return memo[start]

        results = []
        for end in range(start + 1, len(phonemes) + 1):
            sub_phonemes = tuple(phonemes[start:end])
            if sub_phonemes in phoneme_to_words:
                for word in phoneme_to_words[sub_phonemes]:
                    for rest in backtrack(end):
                        results.append([word] + rest)
        memo[start] = results
        return results

    all_combos = backtrack(0)
    unique_combos = {tuple(sorted(combo)) for combo in all_combos}
    return [list(combo) for combo in unique_combos]

if __name__ == "__main__":
    input_seq = ["DH", "EH", "R", "DH", "EH", "R"]
    dict_cmu = load_cmu()
    phoneme_to_words = preprocess_pronunciation_dict(dict_cmu)
    result = find_word_combos_with_pronunciation(input_seq, phoneme_to_words)
    print(result)
    