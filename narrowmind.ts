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

    train(text: string): void {
        // Store raw training text
        this.raw_training_text = text;

        // Split into sentences (., !, ?)
        const sentences = text
            .split(/[.!?]/)
            .map(s => s.trim())
            .filter(s => s.length > 0);

        // Process sentence-level contexts
        for (const sentence of sentences) {
            const sentence_tokens = this.tokenize(sentence);
            if (sentence_tokens.length === 0) continue;

            const context_idx = this.contexts.length;
            this.contexts.push({ tokens: sentence_tokens.slice(), _text: sentence });

            // Map each token to contexts
            for (let word_pos = 0; word_pos < sentence_tokens.length; word_pos++) {
            const token = sentence_tokens[word_pos];
            const word = this.extractWord(token).toLowerCase();

            // Add to word_to_contexts
            if (!this.word_to_contexts.has(word))
                this.word_to_contexts.set(word, []);
            this.word_to_contexts.get(word)!.push(context_idx);

            // Build before/after window (max ±5)
            const before = sentence_tokens.slice(Math.max(0, word_pos - 5), word_pos);
            const after = sentence_tokens.slice(word_pos + 1, word_pos + 6);

            if (!this.context_windows.has(word))
                this.context_windows.set(word, []);
            this.context_windows.get(word)!.push([before, after]);
            }
        }

        // Tokenize all and filter out question words
        const all_tokens = this.tokenize(text);
        const tokens = all_tokens.filter(t => !this.isQuestionWord(this.extractWord(t)));

        // Train n-grams
        for (const ngram_size of this.ngram_sizes) {
            if (tokens.length < ngram_size) continue;

            const ngram_contexts = this.multi_ngram_contexts.get(ngram_size);
            if (!ngram_contexts) continue;

            for (let i = 0; i <= tokens.length - ngram_size; i++) {
            const ngram = tokens.slice(i, i + ngram_size);
            const ngramKey = ngram.join(" ");

            // Count n-grams
            if (ngram_size === this.n) {
                this.ngram_counts.set(
                ngramKey,
                (this.ngram_counts.get(ngramKey) || 0) + 1
                );
            }

            // Build context -> next token
            if (i + ngram_size < tokens.length) {
                const context = ngram.slice(0, ngram_size - 1).join(" ");
                const next_token = tokens[i + ngram_size];

                if (!ngram_contexts.has(context))
                ngram_contexts.set(context, []);

                const continuations = ngram_contexts.get(context)!;
                const found = continuations.find(([tok]) => tok === next_token);
                if (found) {
                found[1] += 1;
                } else {
                continuations.push([next_token, 1]);
                }
            }
            }
        }

        // Sync primary contexts
        const primary = this.multi_ngram_contexts.get(this.n);
        if (primary) this.ngram_contexts = new Map(primary);

        // Vocabulary + unigram counts
        for (const token of tokens) {
            if (!this.vocabulary.includes(token)) this.vocabulary.push(token);
            this.unigram_counts.set(token, (this.unigram_counts.get(token) || 0) + 1);
            this.total_unigrams += 1;
        }

        // Compute TF-IDF
        this.computeTfidf();
    }
    

}

export default NarrowMind;
