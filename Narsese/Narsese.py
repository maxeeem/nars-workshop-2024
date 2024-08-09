from enum import Enum
from typing import Type, Callable
from functools import reduce
from operator import mul

class Config:
    f: float = 1.0 # frequency
    c: float = 0.9 # confidence
    k: int = 1 # evidential horizon

class TermType(Enum):
    ATOM = 0
    STATEMENT = 1

class Copula(Enum):
    Inheritance = "-->"
    Similarity = "<->"

    def __str__(self):
        return str(self.value)

class Punctuation(Enum):
    Judgement = r"."

class Truth:
    def __init__(self, f, c, k):
        self.f = f 
        self.c = c
        self.k = k

    def __repr__(self) -> str:
        return f'%{self.f:.2f};{self.c:.2f}%'

class Term:
    type = TermType.ATOM
    
    def __init__(self, word):
        self.word = word
        
    @property
    def terms(self):
        return (self, )
    
    @property
    def is_atom(self):
        return self.type == TermType.ATOM

    @property
    def is_statement(self):
        return self.type == TermType.STATEMENT

    def __hash__(self) -> int:
        return hash(self.word)
    
    def __eq__(self, o: Type['Term']) -> bool:
        return hash(o) == hash(self)

    def __str__(self) -> str:
        return self.word

class Statement(Term):
    type = TermType.STATEMENT
    
    def __init__(self, subject: Term, copula: Copula, predicate: Term):
        word = "<"+str(subject)+" "+str(copula.value)+" "+str(predicate)+">"
        super().__init__(word)
        self.subject = subject
        self.copula = copula
        self.predicate = predicate

    @property
    def terms(self):
        return (self.subject, self.predicate)

    def __repr__(self) -> str:
        return  f'Statement: {self.word}'

class Sentence:
    truth: Truth = None

    def __init__(self, term: Term, punct: Punctuation):
        self.term = term
        self.word = term.word + str(punct.value)
        self.punct = punct

    def __hash__(self) -> int:
        return hash((self.term, self.punct, self.truth))

    def __str__(self) -> str:
        return self.word

class Task:    
    def __init__(self, sentence: Sentence):
        self.sentence: Sentence = sentence

    @property
    def term(self) -> Term:
        return self.sentence.term
    
    @property
    def truth(self) -> Truth:
        return self.sentence.truth
    
    def __str__(self) -> str:
        return f'{self.sentence.repr()}'

    def __repr__(self) -> str:
        return str(self)

class Judgement(Sentence):
    def __init__(self, term: Term, truth: Truth = None):
        Sentence.__init__(self, term, Punctuation.Judgement)
        self.truth = truth if truth is not None else Truth(Config.f, Config.c, Config.k)

    def __str__(self) -> str:
        return f'{self.word} {self.truth}'


TruthFunction = Callable[[Truth, Truth], Truth]

And = lambda *x: reduce(mul, x, 1)
Or  = lambda *x: 1 - reduce(mul, (1 - xi for xi in x), 1)

w_to_f          = lambda w_plus, w: w_plus/w
w_to_c          = lambda w, k     : w/(w+k)

def truth_from_w(w_plus, w, k):
    f, c = (w_to_f(w_plus, w), w_to_c(w, k)) if w != 0 else (0.5, 0.0)
    return Truth(f, c, k)

F_deduction = lambda f1, c1, f2, c2: (And(f1, f2), And(f1, f2, c1, c2))  # return: f, c
Truth_deduction: TruthFunction = lambda truth1, truth2: Truth(
    *F_deduction(truth1.f, truth1.c, truth2.f, truth2.c), truth1.k)

F_abduction = lambda f1, c1, f2, c2: (And(f1, f2, c1, c2), And(f1, c1, c2))  # return: w+, w
Truth_abduction: TruthFunction = lambda truth1, truth2: truth_from_w(
    *F_abduction(truth1.f, truth1.c, truth2.f, truth2.c), truth1.k)

F_induction = lambda f1, c1, f2, c2: (And(f1, f2, c1, c2), And(f2, c1, c2))  # return: w+, w
Truth_induction: TruthFunction = lambda truth1, truth2: truth_from_w(
    *F_induction(truth1.f, truth1.c, truth2.f, truth2.c), truth1.k)

F_exemplification = lambda f1, c1, f2, c2: (And(f1, f2, c1, c2), And(f1, f2, c1, c2))  # return: w+, w
Truth_exemplification: TruthFunction = lambda truth1, truth2: truth_from_w(
    *F_exemplification(truth1.f, truth1.c, truth2.f, truth2.c), truth1.k)

F_analogy = lambda f1, c1, f2, c2: (And(f1, f2), And(f2, c1, c2))  # return: f, c
Truth_analogy: TruthFunction = lambda truth1, truth2: Truth(
    *F_analogy(truth1.f, truth1.c, truth2.f, truth2.c), truth1.k)

F_resemblance = lambda f1, c1, f2, c2: (And(f1, f2), And(Or(f1, f2), c1, c2))  # return: f, c
Truth_resemblance: TruthFunction = lambda truth1, truth2: Truth(
    *F_resemblance(truth1.f, truth1.c, truth2.f, truth2.c), truth1.k)

F_comparison = lambda f1, c1, f2, c2: (And(f1, f2, c1, c2), And(Or(f1, f2), c1, c2))  # return: w+, w
Truth_comparison: TruthFunction = lambda truth1, truth2: truth_from_w(
    *F_comparison(truth1.f, truth1.c, truth2.f, truth2.c), truth1.k)
