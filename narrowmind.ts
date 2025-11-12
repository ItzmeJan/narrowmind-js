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

    computeTfidf(): void {
        this.total_sentences = this.contexts.length;
        if (this.total_sentences === 0) return;

        // Step 1: Document Frequency (DF)
        const document_frequency = new Map<string, number>();

        for (const context of this.contexts) {
            const sentence_words = new Set<string>();
            for (const token of context.tokens) {
            const word = this.extractWord(token).toLowerCase();
            if (!this.isQuestionWord(word)) {
                sentence_words.add(word);
            }
            }
            for (const word of sentence_words) {
            document_frequency.set(word, (document_frequency.get(word) || 0) + 1);
            }
        }

        // Step 2: Compute IDF = log(total_sentences / (1 + df))
        for (const [word, df] of document_frequency.entries()) {
            const idf = Math.log(this.total_sentences / (1 + df));
            this.idf_scores.set(word, idf);
        }

        // Step 3: Compute TF-IDF per sentence
        this.tfidf_vectors = [];
        for (const context of this.contexts) {
            const tfidf_vector = new Map<string, number>();
            const term_frequency = new Map<string, number>();
            let total_words = 0;

            for (const token of context.tokens) {
            const word = this.extractWord(token).toLowerCase();
            if (!this.isQuestionWord(word)) {
                term_frequency.set(word, (term_frequency.get(word) || 0) + 1);
                total_words++;
            }
            }

            // TF-IDF = (count / total_words) * IDF
            for (const [word, count] of term_frequency.entries()) {
            const tf = count / total_words;
            const idf = this.idf_scores.get(word) ?? 0.0;
            tfidf_vector.set(word, tf * idf);
            }

            this.tfidf_vectors.push(tfidf_vector);
        }
    }
    // returns array of [contextIndex, similarityScore]
    find_similar_contexts_tfidf(query_words: string[]): [number, number][] {
        if (this.tfidf_vectors.length === 0 || query_words.length === 0) return [];

        // Build query TF map
        const query_tf = new Map<string, number>();
        const query_word_count = query_words.length;
        for (const w of query_words) {
            const normalized = this.extract_word(w).toLowerCase();
            if (!this.is_question_word(normalized)) {
            query_tf.set(normalized, (query_tf.get(normalized) || 0) + 1);
            }
        }

        // Build query TF-IDF vector
        const query_vector = new Map<string, number>();
        for (const [word, count] of query_tf.entries()) {
            const tf = count / query_word_count;
            const idf = this.idf_scores.get(word) ?? 0;
            query_vector.set(word, tf * idf);
        }

        // Compare with sentence vectors
        const similarities: [number, number][] = [];
        this.tfidf_vectors.forEach((sentence_vector, idx) => {
            const sim = this.cosine_similarity(query_vector, sentence_vector);
            if (sim > 0) similarities.push([idx, sim]);
        });

        // Sort desc and take top 30
        similarities.sort((a, b) => b[1] - a[1]);
        return similarities.slice(0, 30);
    }

    // cosine similarity helper (using your snake_case name)
    cosine_similarity(vecA: Map<string, number>, vecB: Map<string, number>): number {
        let dot = 0;
        let normA = 0;
        let normB = 0;

        for (const [k, aVal] of vecA.entries()) {
            const bVal = vecB.get(k) ?? 0;
            dot += aVal * bVal;
            normA += aVal * aVal;
        }

        for (const bVal of vecB.values()) {
            normB += bVal * bVal;
        }

        const denom = Math.sqrt(normA) * Math.sqrt(normB);
        return denom === 0 ? 0 : dot / denom;
    }
    compute_tfidf_relevance(candidate_word: string, context_words: string[]): number {
        if (context_words.length === 0 || this.tfidf_vectors.length === 0) {
            return 1.0; // No boost if no context or TF-IDF data
        }

        const context_vector = new Map<string, number>();
        const context_tf = new Map<string, number>();
        const context_word_count = context_words.length;

        // Build context TF
        for (const word of context_words) {
            const normalized_word = this.extract_word(word).toLowerCase();
            if (!this.is_question_word(normalized_word)) {
            context_tf.set(normalized_word, (context_tf.get(normalized_word) || 0) + 1);
            }
        }

        // Compute context TF-IDF vector
        for (const [word, count] of context_tf.entries()) {
            const tf = count / context_word_count;
            const idf = this.idf_scores.get(word) ?? 0;
            context_vector.set(word, tf * idf);
        }

        return this.compute_tfidf_relevance_with_vector(candidate_word, context_vector);
    }

    compute_tfidf_relevance_with_vector(candidate_word: string, context_vector: Map<string, number>): number {
        if (this.tfidf_vectors.length === 0) {
            return 1.0;
        }

        const candidate_word_lower = this.extract_word(candidate_word).toLowerCase();
        const candidate_idf = this.idf_scores.get(candidate_word_lower) ?? 0;

        let total_similarity = 0;
        let sentence_count = 0;

        // Sentences containing candidate word
        const sentence_indices = this.word_to_contexts.get(candidate_word_lower);
        if (sentence_indices) {
            for (const sentence_idx of sentence_indices.slice(0, 10)) {
            const sentence_vector = this.tfidf_vectors[sentence_idx];
            if (sentence_vector) {
                const similarity = this.cosine_similarity(context_vector, sentence_vector);
                if (similarity > 0) {
                total_similarity += similarity;
                sentence_count++;
                }
            }
            }
        }

        const avg_similarity = sentence_count > 0 ? total_similarity / sentence_count : 0;
        const normalized_idf = Math.min(Math.max(candidate_idf / 6.0, 0.0), 1.0);

        const similarity_boost = 1.0 + avg_similarity;      // 1.0–2.0
        const idf_boost = 1.0 + normalized_idf * 0.5;       // 1.0–1.5

        return similarity_boost * idf_boost;                // up to ~3.5x
    }
    tokenize(text: string): string[] {
        const tokens: string[] = [];
        const textLower = text.toLowerCase();
        let currentWord = "";

        for (const ch of textLower) {
            if (/\s/.test(ch)) {
            if (currentWord.length > 0) {
                tokens.push(currentWord);
                currentWord = "";
            }
            } else if (/[a-z0-9]/.test(ch) || ch === "'") {
            // Include apostrophes for contractions
            currentWord += ch;
            } else {
            // Punctuation
            if (currentWord.length > 0) {
                currentWord += ch;
                tokens.push(currentWord);
                currentWord = "";
            } else {
                // Punctuation at start
                tokens.push(ch);
            }
            }
        }

        // Add leftover word
        if (currentWord.length > 0) {
            tokens.push(currentWord);
        }

        return tokens;
    }
    

}

export default NarrowMind;
