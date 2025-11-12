type NGram = string[];
type NGramCount = Map<string, number>;
type NGramContext = Map<string, [string, number][]>;

type ContextEntry = {
  tokens: string[];
  _text: string;
};

class NarrowMind {
  n: number; // Primary n-gram size
  ngram_sizes: number[]; // All n-gram sizes (e.g., [2, 3, 4])
  ngram_weights: Map<number, number>; // Weight for each n-gram
  ngram_counts: NGramCount; // Combined counts
  ngram_contexts: NGramContext; // Primary n-gram contexts
  multi_ngram_contexts: Map<number, NGramContext>; // Per-size contexts
  vocabulary: string[];
  unigram_count: Map<string, number>;
  total_unigrams: number;
  temperature: number;
  top_k: number;
  contexts: ContextEntry[];
  word_to_contexts: Map<string, number[]>;
  context_windows: Map<string, [string[], string[]][]>;
  raw_training_text: string;
  tfidf_vectors: Map<string, number>[];
  idf_scores: Map<string, number>;
  total_sentences: number;

  constructor() {
    this.n = 3;
    this.ngram_sizes = [2, 3];
    this.ngram_weights = new Map([
      [3, 0.5],
      [2, 0.3],
    ]);

    // Normalize weights
    const totalWeight = Array.from(this.ngram_weights.values()).reduce((a, b) => a + b, 0);
    for (const [key, val] of this.ngram_weights) {
      this.ngram_weights.set(key, val / totalWeight);
    }

    this.ngram_counts = new Map();
    this.ngram_contexts = new Map();

    this.multi_ngram_contexts = new Map();
    for (const size of this.ngram_sizes) {
      this.multi_ngram_contexts.set(size, new Map());
    }

    this.vocabulary = [];
    this.unigram_count = new Map();
    this.total_unigrams = 0;
    this.temperature = 1.0;
    this.top_k = 40;

    this.contexts = [];
    this.word_to_contexts = new Map();
    this.context_windows = new Map();

    this.raw_training_text = "";
    this.tfidf_vectors = [];
    this.idf_scores = new Map();
    this.total_sentences = 0;
  }

  train(data: string): void {
    this.raw_training_text = data;
    console.log("Training on data length:", data.length);
    // Training logic will go here
  }
}

export default NarrowMind;
