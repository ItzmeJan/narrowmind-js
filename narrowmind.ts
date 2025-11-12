type NGram = string[];
type NGramCount = Map<string, number>;
type NGramContext = Map<string, [string, number][]>;

type ContextEntry = {
  tokens: string[];
  _text: string;
};

class NarrowMind {
  n: number;
  ngram_sizes: number[];
  ngram_weights: Map<number, number>;
  ngram_counts: NGramCount;
  ngram_contexts: NGramContext;
  multi_ngram_contexts: Map<number, NGramContext>;
  vocabulary: string[];
  unigram_counts: Map<string, number>;
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

    const totalWeight = Array.from(this.ngram_weights.values()).reduce((a, b) => a + b, 0);
    for (const [key, val] of this.ngram_weights.entries()) {
      this.ngram_weights.set(key, val / totalWeight);
    }

    this.ngram_counts = new Map();
    this.ngram_contexts = new Map();

    this.multi_ngram_contexts = new Map();
    for (const size of this.ngram_sizes) {
      this.multi_ngram_contexts.set(size, new Map());
    }

    this.vocabulary = [];
    this.unigram_counts = new Map();
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

  static newWithNgrams(ngram_sizes: number[], weights: number[]): NarrowMind {
    if (ngram_sizes.length !== weights.length) {
      throw new Error("ngram_sizes and weights must have the same length");
    }

    const primary_n = ngram_sizes.length ? Math.max(...ngram_sizes) : 3;
    const ngram_weights = new Map<number, number>();

    for (let i = 0; i < ngram_sizes.length; i++) {
      ngram_weights.set(ngram_sizes[i], weights[i]);
    }

    const total_weight = Array.from(ngram_weights.values()).reduce((a, b) => a + b, 0);
    if (total_weight > 0) {
      for (const [size, weight] of ngram_weights.entries()) {
        ngram_weights.set(size, weight / total_weight);
      }
    }

    const multi_ngram_contexts = new Map<number, NGramContext>();
    for (const size of ngram_sizes) {
      multi_ngram_contexts.set(size, new Map());
    }

    const model = new NarrowMind();
    model.n = primary_n;
    model.ngram_sizes = ngram_sizes;
    model.ngram_weights = ngram_weights;
    model.multi_ngram_contexts = multi_ngram_contexts;
    return model;
  }

  train(data: string): void {
    this.raw_training_text = data;
    console.log("Training on data length:", data.length);
  }
}

export default NarrowMind;
