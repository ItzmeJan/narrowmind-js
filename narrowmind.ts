type NGram = string[];
type NGramCount = Map<NGram, number>;
type NGramContext = Map<string[], [string, number][]>;

type ContextEntry = {
    tokens: string[],
    _text: string,
}

class NarrowMind {
    tokens:string[]
    
    constructor() {
        this.tokens = [];

    }

    train(data:string) {
        
    }
}

module.exports =  NarrowMind;