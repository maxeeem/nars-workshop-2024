from collections import defaultdict

from Narsese.Narsese import *

from Narsese.narsese_lark import Lark_StandAlone, Transformer, v_args, Token

inline_args = v_args(inline=True)

class TreeToNarsese(Transformer):
    
    f: float = 1.0
    c_judgement: float = 0.9
    k: int = 1

    @inline_args
    def task(self, *args):
        kwargs = dict(args)
        return Task(**kwargs)

    @inline_args
    def judgement(self, statement: 'Term|Statement', *args):
        kwargs = dict(args)
        truth = kwargs.pop('truth', None)
        if truth is not None:
            f, c, k = truth
            if c is None:
                c = self.c_judgement
        else:
            f, c, k = self.f, self.c_judgement, self.k

        kwargs['truth'] = Truth(f,c,k)
        return ('sentence', Judgement(statement, **kwargs))

    @inline_args
    def statement(self, term1, copula, term2):
        return Statement(term1, copula, term2)
        
    @inline_args
    def truth(self, f: Token, c: Token=None, k: Token=None):
        # truth : "%" frequency [";" confidence] "%"
        f = float(f.value)
        c = float(c.value) if c is not None else None
        k = float(k.value) if k is not None else self.k
        return ('truth',(f, c, k))

    @inline_args
    def atom_term(self, word: Token):
        word = word.value
        return Term(word)

    @inline_args
    def inheritance(self):
        return Copula.Inheritance
    
    @inline_args
    def similarity(self):
        return Copula.Similarity

def parse(narsese: str, rule=False):
    tree = TreeToNarsese()
    parser = Lark_StandAlone(transformer=tree)
    task = parser.parse(narsese)
    return task.term if rule else task.sentence
