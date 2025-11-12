type NGram = string[];
type NGramCount = Map<NGram, number>;
type NGramContext = Map<string[], [string, number][]>;

type ContextEntry = {
    tokens: string[],
    _text: string,
}

class NarrowMind {
    n!: number // Primary n-gram size (for backward compatibility)
    ngram_sizes!: number[] // All n-gram sizes to train (e.g., [2, 3, 4])
    ngram_weights!: Map<number, number> // Weights for each n-gram size (must sum to ~1.0)
    ngram_counts!: NGramCount // Combined counts (for backward compatibility)
    ngram_contexts!: NGramContext // Primary n-gram contexts (for backward compatibility)
    multi_ngram_contexts!: Map<number, NGramContext> // Separate contexts for each n-gram size
    vocabulary!: string[]
    unigram_count!: Map<string, number> // For smoothing and backoff
    total_unigrams!: number // Total word count for probability calculations
    temperature!: number // Controls randomness: 1.0 = normal, <1.0 = more deterministic, >1.0 = more random
    top_k!: number // Only consider top-k most likely tokens (0 = no limit)
    // Full text context scanning
    contexts!: ContextEntry[] // All sentence-level contexts from training data
    word_to_contexts!: Map<string, number[]> // Maps words to context indices where they appear
    context_windows!: Map<string, [string[], string[]][]> // Word -> (before_context, after_context) pairs
    raw_training_text!: string // Raw training text for direct pattern matching
    // TF-IDF for vector-based sentence selection
    tfidf_vectors!: Map<string, number>[] // TF-IDF vectors for each sentence
    idf_scores!: Map<String, number> // Inverse document frequency for each word
    total_sentences!: number // Total number of sentences for IDF calculation

    constructor() {
        

    }

    train(data:string) {
        
    }
}

module.exports =  NarrowMind;